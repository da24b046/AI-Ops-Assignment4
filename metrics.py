from prometheus_client import (
    Counter, Gauge, Histogram, Summary, start_http_server
)

IMAGES = Counter(
    "caption_images_processed_total",
    "Successfully captioned images",
    ["mode"]
)

REQUESTS = Counter(
    "caption_requests_total",
    "Captioning jobs by outcome",
    ["mode", "status"]
)

ERRORS = Counter(
    "caption_errors_total",
    "Captioning errors by category",
    ["mode", "error_type"]
)

ACTIVE = Gauge(
    "caption_active_requests",
    "Currently running caption jobs",
    ["mode"]
)

INFERENCE = Histogram(
    "caption_inference_seconds",
    "Per-image CPU inference duration",
    ["mode"],
    buckets=(0.25, 0.5, 1, 2, 3, 5, 10, 20, 40, 60)
)

JOB_TIME = Summary(
    "caption_job_duration_seconds",
    "Total job execution duration",
    ["mode"]
)

BATCH_SIZE = Histogram(
    "caption_batch_size",
    "Number of images submitted per job",
    buckets=(1, 2, 5, 10, 20, 50)
)

INPUT_BYTES = Counter(
    "caption_input_bytes_total",
    "Bytes in uploaded images",
    ["mode"]
)

LAST_INFERENCE = Gauge(
    "caption_last_inference_seconds",
    "Most recent successful inference duration"
)

MODEL_LOADED = Gauge(
    "caption_model_loaded",
    "One when the model is loaded successfully"
)


def start_metrics():
    start_http_server(8000, addr="127.0.0.1")
    return True
