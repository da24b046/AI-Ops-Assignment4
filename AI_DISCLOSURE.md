# AI Disclosure

AI tools (ChatGPT) were used to assist with the development of this assignment, including:

- Structuring the Streamlit application and Prometheus instrumentation.
- Generating Grafana dashboard configurations and PromQL queries.
- Assisting with Alertmanager setup, debugging, and documentation.

**PromQL queries** were reviewed and tested against actual application and system metrics. The corresponding query logic and calculations were verified during evaluation.

The application was implemented, executed, tested, and validated locally on Ubuntu. AI was used as a development aid, while responsibility for the final implementation and submission remains with the student.

# AI Attribution — PromQL Queries

**Tool Used:** ChatGPT (OpenAI)

**Prompt Used:**

"Generate PromQL queries for monitoring a CPU-only BLIP image-captioning application using Prometheus and Node Exporter. Include image throughput, average inference latency, P95 inference latency, request success and failure rates, CPU utilization, memory utilization, disk usage, and correlations between batch inference and system resource consumption."

AI assistance was used to construct and explain PromQL queries and assist with dashboard configuration. The queries were reviewed and tested against actual application and system metrics collected during single-image and 30-image bulk processing experiments.

The metric calculations were manually verified, and the final implementation was tested locally on Ubuntu.

