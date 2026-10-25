import time
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_NAME = "Salesforce/blip-image-captioning-base"

# Force CPU-only operation, as required by Assignment 4.
DEVICE = torch.device("cpu")
torch.set_num_threads(4)

_processor = None
_model = None


def load_model():
    global _processor, _model

    if _model is None:
        print("[INFO] Loading BLIP model on CPU...", flush=True)

        _processor = BlipProcessor.from_pretrained(MODEL_NAME)

        _model = BlipForConditionalGeneration.from_pretrained(
            MODEL_NAME
        )

        _model.to(DEVICE)
        _model.eval()

        print("[INFO] BLIP model loaded successfully.", flush=True)
        print("[INFO] Device:", next(_model.parameters()).device, flush=True)

    return _processor, _model


def generate_caption(image):
    processor, model = load_model()

    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=3
        )

    duration = time.perf_counter() - start

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption, duration
