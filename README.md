# DA3408 Assignment 4 — AI Image Captioning Observability



## 1. Features

- **Single mode:** Upload a JPEG, PNG, or WebP image; generate and display its caption and measured inference duration.
- **Bulk mode:** Upload a ZIP with up to **40 supported images** (30-image ZIP used for the demonstration); view per-image captions/status/timings, and download a CSV.
- **CPU-only inference:** PyTorch runs BLIP on `torch.device("cpu")`, even if a CUDA GPU is available.
- **Application instrumentation:** Counter, Gauge, Histogram, and Summary metrics with bounded labels; `/metrics` is exposed independently of Streamlit.
- **Infrastructure monitoring:** Node Exporter measures CPU, available memory, filesystem/disk statistics and other host telemetry.
- **Observability and alerts:** Prometheus scrapes at 5-second intervals; Grafana displays the metrics; Prometheus evaluates four alert rules and forwards firing alerts to Alertmanager.

## 2. Architecture

```text
Browser
  │   Upload one image or a ZIP; inspect results
  ▼
Streamlit UI (:8501) ──► BLIP processor/model (CPU)
  │                           │
  └──────────► prometheus_client metrics (:8000/metrics)
                                 │
Ubuntu host ──► Node Exporter (:9100/metrics)
                                 │
                                 ▼
                    Prometheus (:9090)
                 scrapes both + itself every 5s
                       │               │
                       ▼               ▼
               Grafana (:3000)   Alertmanager (:9093)
               dashboard         receives firing alerts
```

Services use `127.0.0.1` for local development. A separate terminal is used for each foreground process; Grafana runs as a systemd service.

### Ports

| Component | Address | Purpose |
|---|---|---|
| Streamlit | `http://localhost:8501` | Application UI |
| Python metrics exporter | `http://localhost:8000/metrics` | Application metrics |
| Node Exporter | `http://localhost:9100/metrics` | Host telemetry |
| Prometheus | `http://localhost:9090` | Metrics, queries and alert evaluation |
| Grafana | `http://localhost:3000` | 16-panel dashboard |
| Alertmanager | `http://localhost:9093` | Alert lifecycle UI |

## 3. Repository contents

```text
AI-Ops-Assignment4/
├── app.py                              # Streamlit single/bulk upload UI
├── model_utils.py                      # CPU-only BLIP loading and inference
├── metrics.py                          # Prometheus metrics and HTTP exporter
├── test_model.py                       # Basic model smoke test
├── requirements.txt                    # Application dependencies
├── prometheus.yml                      # Scrape jobs and Alertmanager endpoint
├── alert_rules.yml                     # Four Prometheus alert rules
├── alertmanager.yml                    # Local demonstration receiver
├── dashboards/
│   └── aiops_caption_monitor.json      # Dashboard JSON deliverable
├── scripts/
│   └── create_dashboard.py             # Dashboard generator (see note below)
├── grafana/
│   └── provisioning/
│       ├── datasources/prometheus.yml
│       └── dashboards/aiops.yml
├── screenshots/                        # Supporting screenshots
├── README.md
└── .gitignore
```

**Not committed:** downloaded monitoring binaries, Prometheus time-series data, Alertmanager runtime data, cached model weights, the local Python environment, bulk test photographs, ZIP archives, temporary CSVs and backups.

**Important:** The generator script may lag behind manual improvements to the exported JSON (particularly panels 11, 14, and 16). Treat `dashboards/aiops_caption_monitor.json` as the **authoritative final dashboard** unless you update and verify the generator produces an identical configuration.

## 4. Environment and dependencies

### Tested software

| Software | Tested version |
|---|---|
| Ubuntu | 26.04.1 LTS, x86_64 |
| Python | 3.14.4 |
| PyTorch | 2.13.0 (CUDA-capable build installed, CPU used for inference) |
| Transformers | 5.20.2 |
| Streamlit | 1.66.1 |
| Pillow | 12.3.0 |
| prometheus-client | 0.26.0 |
| Prometheus | 3.15.1 |
| Node Exporter | 1.12.1 |
| Grafana | 13.2.4 |
| Alertmanager | 0.34.1 |

`torchvision` was **not needed** for the working BLIP path. The Streamlit file watcher was disabled to prevent optional Transformers-module inspection from producing `torchvision` import warnings. Models and package behavior can change in newer environments.

### Installation

The project was tested in an existing virtual environment named `AIops`. To reproduce in a new environment, use a compatible Python interpreter and install the application dependencies:

```bash
cd ~/AI_Ops_Assigments/AI-Ops-Assignment4
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

On other operating systems, Python versions, or GPU configurations, install a compatible PyTorch wheel using [PyTorch's installer](https://pytorch.org/get-started/locally/). Even on a CUDA-capable system, this assignment **runs inference on CPU**. The first model run downloads pretrained weights from Hugging Face; an internet connection and sufficient disk space are required.

Test model loading:

```bash
python test_model.py
```

The test image is synthetic, so its caption may not be semantically accurate; evaluate user-facing caption quality with real photographs.

## 5. Run the application

**Terminal 1 — Streamlit**

```bash
cd ~/AI_Ops_Assigments/AI-Ops-Assignment4
source ~/AIops/bin/activate    # Tested environment; adapt if using .venv
python -m streamlit run app.py \
  --server.port 8501 \
  --server.fileWatcherType none
```

Open `http://localhost:8501`.

- **Single Image:** Choose **Single Image**, upload a photo, click **Generate Caption**, and inspect caption/inference time.
- **Bulk ZIP:** Choose **Bulk ZIP**, upload a ZIP containing supported images, click **Process ZIP**, and download `captions.csv` after completion.

The ZIP path includes checks for supported image extensions, image-size limits, maximum image count, per-entry uncompressed sizes, and total uncompressed size. Individual decoding failures are recorded as errors without necessarily aborting the whole batch. This is a student demonstration rather than a hardened public upload service: use trusted test files only.

Check exporter with **GET** (not `curl -I`, which sends HEAD and may return 405):

```bash
curl -fsS http://localhost:8000/metrics | grep '^caption_' | head -50
```

The BLIP model is **lazy-loaded**. After restarting Streamlit, `caption_model_loaded` is `0` until a caption request loads it, then becomes `1`.

## 6. Metrics and label design

Instrumentation lives in `metrics.py` and is updated from `app.py`.

| Type | Metric family | Meaning | Labels |
|---|---|---|---|
| Counter | `caption_images_processed_total` | Successfully captioned images | `mode` |
| Counter | `caption_requests_total` | Finished jobs by outcome | `mode`, `status` |
| Counter | `caption_errors_total` | Processing error events | `mode`, `error_type` |
| Counter | `caption_input_bytes_total` | Successfully processed uploaded image bytes | `mode` |
| Gauge | `caption_active_requests` | Currently processing jobs | `mode` |
| Gauge | `caption_model_loaded` | 0/1 model initialization indicator | — |
| Gauge | `caption_last_inference_seconds` | Latest successful per-image inference duration | — |
| Histogram | `caption_inference_seconds` | Distribution of per-image CPU inference latency | `mode` |
| Histogram | `caption_batch_size` | Distribution of submitted job sizes | — |
| Summary | `caption_job_duration_seconds` | Overall job duration | `mode` |

**Naming details:** Histogram families expose `_bucket`, `_sum`, `_count`. The Python Summary exposes `_sum` and `_count` (not a `_total` observation series). Counters conventionally expose `_total`.

**Labels:** `mode` is `single` or `bulk`; `status` includes `success` / `error`; `error_type` uses the Python exception class name. These labels have bounded or relatively constrained cardinality. Raw client IPs and unique session IDs were intentionally **not** included in the implemented metrics to avoid privacy exposure and unbounded series cardinality. **Assignment caveat:** The handout explicitly requests client metadata labels (for example source IP/session ID); the current implementation has not demonstrated unique-client tracking. A privacy-preserving, bounded client classification would be a suitable additional implementation if strict coverage of that criterion is needed.

**Counter interpretation:** One successful 30-image ZIP job increases the bulk request counter by **1**, and the bulk image counter by **30**. Counters reset when the Python process restarts; Prometheus handles counter resets in `rate()` and `increase()` queries.

## 7. Start Node Exporter and Prometheus

Download the appropriate Linux AMD64 release binaries from the official [Prometheus downloads](https://prometheus.io/download/) page and place the `prometheus`, `promtool`, and `node_exporter` executables in `monitoring/bin/`.

Validate configs:

```bash
./monitoring/bin/promtool check rules alert_rules.yml
./monitoring/bin/promtool check config prometheus.yml
```

**Terminal 3 — Node Exporter**

```bash
cd ~/AI_Ops_Assigments/AI-Ops-Assignment4
./monitoring/bin/node_exporter --web.listen-address=127.0.0.1:9100
```

**Terminal 4 — Prometheus**

```bash
cd ~/AI_Ops_Assigments/AI-Ops-Assignment4
mkdir -p monitoring/data
./monitoring/bin/prometheus \
  --config.file=prometheus.yml \
  --storage.tsdb.path=monitoring/data \
  --storage.tsdb.retention.time=7d \
  --web.listen-address=127.0.0.1:9090
```

The included `prometheus.yml` is configured for three jobs with a **5-second scrape interval**:

| Job name | Target |
|---|---|
| `ai_captioning` | `127.0.0.1:8000` |
| `node_exporter` | `127.0.0.1:9100` |
| `prometheus` | `127.0.0.1:9090` |

Verify:

```bash
curl -fsSG 'http://127.0.0.1:9090/api/v1/query' \
  --data-urlencode 'query=up' | python -m json.tool
```

All three jobs should report a sample value of `"1"`. View them at `http://localhost:9090/targets`.

**Configuration reload after editing:**

```bash
pgrep -af 'monitoring/bin/prometheus'
# Inspect the PID carefully, then:
kill -HUP ACTUAL_PID_HERE
```

Use the numeric PID, without angle brackets. Avoid killing any unrelated process.

## 8. Grafana setup and dashboard

Install Grafana OSS from its [official installation instructions](https://grafana.com/docs/grafana/latest/setup-grafana/installation/debian/). On the tested system, it runs as a systemd service:

```bash
sudo systemctl enable --now grafana-server
sudo systemctl status grafana-server --no-pager
curl -fsS http://localhost:3000/api/health | python -m json.tool
```

Open `http://localhost:3000` and configure Prometheus as the **default data source**:

- **Name:** `Prometheus`
- **UID:** `aiops-prometheus`
- **Type:** Prometheus
- **URL:** `http://127.0.0.1:9090`
- **Minimum data interval:** `5s`

The repository contains YAML for data-source provisioning (`grafana/provisioning/datasources/prometheus.yml`). Copy it to `/etc/grafana/provisioning/datasources/` and restart Grafana if you want automatic setup.

**Dashboard options:**

1. **Import:** In Grafana, import `dashboards/aiops_caption_monitor.json`, selecting the correctly configured Prometheus data source.
2. **Provision:** Copy `grafana/provisioning/dashboards/aiops.yml` to `/etc/grafana/provisioning/dashboards/aiops.yml`; create `/var/lib/grafana/dashboards/aiops/`; copy the JSON into that directory; restart Grafana. This matches the configured file-provider path.

```bash
sudo mkdir -p /var/lib/grafana/dashboards/aiops
sudo cp dashboards/aiops_caption_monitor.json /var/lib/grafana/dashboards/aiops/
sudo cp grafana/provisioning/dashboards/aiops.yml \
  /etc/grafana/provisioning/dashboards/aiops.yml
sudo systemctl restart grafana-server
```

Direct dashboard URL after import/provisioning: `http://localhost:3000/d/aiops-caption-monitor`. Use **Last 15 minutes** (adjust as needed) and **Refresh: 5s** during the demonstration.

### Dashboard panel inventory (16)

| ID | Panel | Interpretation |
|---:|---|---|
| 01 | Application Health | Scrape reachability (`up`) |
| 02 | Total Images Processed | Sum of successful-image Counters |
| 03 | Model Loaded | BLIP initialization state |
| 04 | Image Throughput | Images per second, by mode |
| 05 | Average Inference Latency | Mean seconds per image, by mode |
| 06 | P95 Inference Latency | Approximate 95th-percentile seconds per image |
| 07 | Request Outcomes | Successful/failed jobs per second |
| 08 | Active Processing Jobs | Concurrent processing jobs by mode |
| 09 | Average Job Duration | Mean full-job duration |
| 10 | Error Events | Errors per minute, by category |
| 11 | Cumulative Batch Size Distribution | Histogram bucket observation rates |
| 12 | CPU Utilization | Host CPU usage percentage |
| 13 | RAM Utilization | Host used-memory percentage estimate |
| 14 | CPU Utilization vs Bulk Throughput | Two real series, separate y-axes (%) and images/sec |
| 15 | Root Disk Utilization | Root filesystem use percentage |
| 16 | Available System Memory | Available bytes, Grafana-formatted |

**Panel 14 warning:** Percent CPU and images/sec have different units; they **must not** be silently multiplied and plotted as though they are the same unit. The final exported JSON uses a separate right axis for bulk throughput.



**Final manual visual check:** Confirm Grafana actually renders these settings in your installed version, especially the distinct axes on Panel 14, readable legends, and graph descriptions. JSON configuration alone does not prove visual compliance.

## 9. PromQL queries and manual calculations

The following examples explain how the dashboard values are derived. Some queries are generated with AI assistance (attribution is below).

### Application reachability

```promql
up{job="ai_captioning"}
```

`1` means Prometheus successfully scraped the application; `0` means the scrape failed. The app can be scrapeable even when BLIP has not loaded, so also check `caption_model_loaded`.

### Image throughput

```promql
sum by (mode) (rate(caption_images_processed_total[1m]))
```

A Counter's rate estimates the **number of successfully processed images per second**, over the one-minute window, separately for `single` and `bulk`. Example: 30 new images in exactly 20 seconds corresponds to a rough workload average of **1.5 images/sec** (not necessarily the same as a Prometheus extrapolated one-minute `rate()`).

### Mean inference latency

```promql
sum by (mode) (rate(caption_inference_seconds_sum[5m]))
/
sum by (mode) (rate(caption_inference_seconds_count[5m]))
```

The numerator represents inference-duration seconds recorded per second; the denominator represents completed images per second. Dividing gives seconds/image. **Observed initial example:** two single images accumulated 1.364061 seconds → **0.682 seconds/image**; 60 bulk images accumulated 38.318279 seconds → **0.639 seconds/image**. These are **whole-run cumulative means**, not necessarily the five-minute moving averages.

When no image is completed in the window, `0/0` can evaluate to `NaN`; this means **no recent observations**, not zero latency.

### P95 latency

```promql
histogram_quantile(
  0.95,
  sum by (le, mode) (rate(caption_inference_seconds_bucket[5m]))
)
```

Prometheus estimates the value below which approximately 95% of recent observations lie, interpolating within classic histogram buckets. This is an **estimate** dependent on bucket boundaries.

### CPU utilization

```promql
100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle"}[1m])))
```

Example: mean idle fraction `0.80` means utilization of `100 × (1 − 0.80) = 20%`, averaged across logical CPU cores. On a machine with 16 logical processors, one saturated core may contribute only about 6.25 percentage points of whole-system utilization.

### Memory utilization

```promql
100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)
```

Example: 5 GiB available out of 14 GiB total corresponds to approximately `100 × (1 − 5/14) = 64.3%` of memory unavailable for new allocations. `MemAvailable` is an operating-system estimate, not simply `MemFree`.

### Disk utilization

```promql
100 * (
  1 - node_filesystem_avail_bytes{mountpoint="/",fstype!~"tmpfs|overlay"}
      / node_filesystem_size_bytes{mountpoint="/",fstype!~"tmpfs|overlay"}
)
```

Example: 50 GiB available out of 144 GiB total implies approximately 65.3% unavailable. Filesystem statistics depend on the host mount and reserved blocks; inspect labels to confirm the expected root partition.

### Job outcomes and errors

```promql
sum by (mode, status) (rate(caption_requests_total[5m]))
```

Gives **jobs/sec**, not images/sec. For error-event rate per minute:

```promql
sum by (mode, error_type) (increase(caption_errors_total[5m])) / 5
```

Dividing the estimated five-minute increase by five converts it to a **five-minute average per minute**, not an instantaneous event count.

### Bulk activity versus CPU utilization

Overlay:

```promql
100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle"}[1m])))
```

with:

```promql
sum(rate(caption_images_processed_total{mode="bulk"}[1m]))
```

Display CPU in **% on the left axis**, and bulk image rate in **images/sec on the right**. Coincident changes suggest a relationship, but they do not by themselves prove causation.

### Model and system checks

```promql
caption_model_loaded
node_memory_MemAvailable_bytes
sum by (mode) (caption_active_requests)
```

Metric values change only when the application acts or Prometheus refreshes its scrape. No synthetic observations are inserted merely to populate the dashboard.

## 10. Alerting configuration and test

Four rules are in `alert_rules.yml`:

| Rule | PromQL condition | Required duration |
|---|---|---|
| `AICaptioningAppDown` | `up{job="ai_captioning"} == 0` | 20 seconds |
| `HighCPUUsage` | Mean host CPU busy percentage > 80 | 1 minute |
| `HighMemoryUsage` | Host memory utilization > 85% | 1 minute |
| `SlowInference` | Mean inference > 3 s with at least 5 recent images | 30 seconds |

Validate:

```bash
./monitoring/bin/promtool check rules alert_rules.yml
./monitoring/bin/promtool check config prometheus.yml
./monitoring/bin/amtool check-config alertmanager.yml
```

Install the official Linux AMD64 Alertmanager binaries (`alertmanager` and `amtool`) under `monitoring/bin/`.

**Terminal 5 — Alertmanager**

```bash
cd ~/AI_Ops_Assigments/AI-Ops-Assignment4
./monitoring/bin/alertmanager \
  --config.file=alertmanager.yml \
  --storage.path=monitoring/alertmanager-data \
  --web.listen-address=127.0.0.1:9093
```

The included `prometheus.yml` forwards alerts to `127.0.0.1:9093`. Check connectivity:

```bash
curl -fsS http://localhost:9093/-/ready
curl -fsS http://localhost:9090/api/v1/alertmanagers | python -m json.tool
```

**Receiver limitation:** The `local-demo` route has no email/Slack/webhook integration. It supports demonstration and inspection of alert state in Alertmanager, **not external notifications**.

### Controlled downtime demonstration

1. Ensure Streamlit, Node Exporter, Prometheus and Alertmanager are running.
2. Stop **only** Streamlit using `Ctrl+C` in its terminal.
3. Verify `up{job="ai_captioning"}` becomes `0` after a scrape.
4. After the configured `for` duration, check that `AICaptioningAppDown` reaches `firing` in `http://localhost:9090/alerts` and appears in `http://localhost:9093`.
5. Capture the firing-alert screenshot for the submission evidence.
6. Restart Streamlit with the command in Section 5; verify `up` returns to `1` and alert lists eventually become empty.
7. Generate a caption to change `caption_model_loaded` from `0` back to `1`.

**Observed demonstration:** Alertmanager was connected and a firing alert screenshot was captured; after restoration, Prometheus reported `up=1`, `caption_model_loaded=1`, and empty active-alert lists. Avoid overloading your laptop solely to force CPU/RAM alerts.

## 11. Experiments, measurements and expected results

### A. Single-image test

Upload a real photograph, generate a caption and observe a successful job/image increment. Capture the image, caption and inference duration as proof.

### B. Bulk-image workload

Use a ZIP of **30 real photographs**. Verify results for each file and a CSV export. This provides sustained CPU inference activity and should produce measurable throughput/latency/CPU changes.

Run several 30-image batches **after** Prometheus begins scraping to populate historical graphs. Avoid assuming CPU must reach any fixed percentage: total usage is averaged across 16 logical processors and depends on other system work.

### C. Initial verified application data

The initial successful test showed:

| Measure | Single | Bulk |
|---|---:|---:|
| Successfully captioned images | 2 | 60 |
| Successful jobs | 2 | 2 |
| Total measured inference time | 1.364061 s | 38.318279 s |
| Approx. mean inference per image | 0.682 s | 0.639 s |
| Total job duration | 3.811848 s | 38.760611 s |

These values were **observed before later restarts and experiments**. They are not claims about a permanent benchmark or current live counters.





