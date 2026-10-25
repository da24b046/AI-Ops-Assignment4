import csv
import io
import time
import zipfile

import streamlit as st
from PIL import Image, UnidentifiedImageError

from model_utils import generate_caption, load_model
from metrics import (
    IMAGES, REQUESTS, ERRORS, ACTIVE, INFERENCE,
    JOB_TIME, BATCH_SIZE, INPUT_BYTES,
    LAST_INFERENCE, MODEL_LOADED, start_metrics
)

st.set_page_config(
    page_title="AI Image Captioning Monitor",
    page_icon="📷",
    layout="wide"
)

MAX_IMAGES = 40
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


@st.cache_resource
def initialize_metrics():
    return start_metrics()


@st.cache_resource(show_spinner="Loading BLIP model on CPU...")
def initialize_model():
    processor, model = load_model()
    MODEL_LOADED.set(1)
    return processor, model


def caption_one(data, mode):
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds 10 MB limit")

    with Image.open(io.BytesIO(data)) as original:
        if original.width * original.height > MAX_PIXELS:
            raise ValueError("Image dimensions are too large")
        image = original.convert("RGB")

    caption, seconds = generate_caption(image)

    IMAGES.labels(mode=mode).inc()
    INFERENCE.labels(mode=mode).observe(seconds)
    LAST_INFERENCE.set(seconds)
    INPUT_BYTES.labels(mode=mode).inc(len(data))

    return caption, seconds


def process_single(upload):
    mode = "single"
    start = time.perf_counter()
    ACTIVE.labels(mode=mode).inc()

    try:
        initialize_model()
        caption, seconds = caption_one(upload.getvalue(), mode)
        BATCH_SIZE.observe(1)
        REQUESTS.labels(mode=mode, status="success").inc()

        st.image(upload, width=450)
        st.success(f"Caption: {caption}")
        st.info(f"CPU inference time: {seconds:.2f} seconds")

    except Exception as exc:
        REQUESTS.labels(mode=mode, status="error").inc()
        ERRORS.labels(
            mode=mode,
            error_type=type(exc).__name__
        ).inc()
        st.error(f"Processing failed: {exc}")

    finally:
        JOB_TIME.labels(mode=mode).observe(
            time.perf_counter() - start
        )
        ACTIVE.labels(mode=mode).dec()


def process_bulk(upload):
    mode = "bulk"
    start = time.perf_counter()
    ACTIVE.labels(mode=mode).inc()

    try:
        data = upload.getvalue()

        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = [
                item for item in archive.infolist()
                if not item.is_dir()
                and not item.filename.startswith("__MACOSX/")
                and item.filename.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".webp")
                )
            ]

            if not entries:
                raise ValueError("ZIP contains no supported images")

            if len(entries) > MAX_IMAGES:
                raise ValueError(
                    f"Maximum {MAX_IMAGES} images per ZIP"
                )

            if any(item.file_size > MAX_IMAGE_BYTES for item in entries):
                raise ValueError("An image exceeds the 10 MB limit")

            if sum(item.file_size for item in entries) > 100 * 1024 * 1024:
                raise ValueError("ZIP uncompressed images exceed 100 MB")

            initialize_model()
            BATCH_SIZE.observe(len(entries))
            results = []
            progress = st.progress(0)

            for index, item in enumerate(entries):
                try:
                    image_data = archive.read(item)
                    caption, seconds = caption_one(image_data, mode)
                    results.append({
                        "filename": item.filename,
                        "caption": caption,
                        "seconds": round(seconds, 3),
                        "status": "success"
                    })
                except Exception as exc:
                    ERRORS.labels(
                        mode=mode,
                        error_type=type(exc).__name__
                    ).inc()
                    results.append({
                        "filename": item.filename,
                        "caption": "",
                        "seconds": 0,
                        "status": "error"
                    })

                progress.progress((index + 1) / len(entries))

        successes = sum(r["status"] == "success" for r in results)
        status = "success" if successes == len(results) else "error"
        REQUESTS.labels(mode=mode, status=status).inc()

        st.write(f"Successfully captioned {successes}/{len(results)} images")
        st.dataframe(results, use_container_width=True)

        csv_buffer = io.StringIO()
        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=["filename", "caption", "seconds", "status"]
        )
        writer.writeheader()
        writer.writerows(results)

        st.download_button(
            "Download captions CSV",
            data=csv_buffer.getvalue(),
            file_name="captions.csv",
            mime="text/csv"
        )

    except Exception as exc:
        REQUESTS.labels(mode=mode, status="error").inc()
        ERRORS.labels(
            mode=mode,
            error_type=type(exc).__name__
        ).inc()
        st.error(f"Bulk processing failed: {exc}")

    finally:
        JOB_TIME.labels(mode=mode).observe(
            time.perf_counter() - start
        )
        ACTIVE.labels(mode=mode).dec()


initialize_metrics()

st.title("📷 CPU Image Captioning & Observability")
st.caption("DA3408 Assignment 4 | BLIP CPU inference | Prometheus Monitoring")

single_tab, bulk_tab = st.tabs(["Single Image", "Bulk ZIP"])

with single_tab:
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="single"
    )

    if uploaded_image and st.button("Generate Caption"):
        process_single(uploaded_image)

with bulk_tab:
    uploaded_zip = st.file_uploader(
        "Upload ZIP of images (maximum 40)",
        type=["zip"],
        key="bulk"
    )

    if uploaded_zip and st.button("Process ZIP"):
        process_bulk(uploaded_zip)

st.divider()
st.caption("Application metrics: http://localhost:8000/metrics")
