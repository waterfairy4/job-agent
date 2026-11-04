# Marco Rossi (fictional candidate) — project notes

## LLM inference serving
- Moved a 13B chat model from a naive Hugging Face server to vLLM with continuous batching and PagedAttention; throughput rose 3.1x on the same GPUs.
- Compared AWQ and GPTQ 4-bit quantization: AWQ kept quality within 1 point on an internal benchmark while cutting GPU memory by 58%.
- Kubernetes autoscaling on queue length; P99 latency kept under 2.5 seconds at peak traffic.
- Monitoring with Prometheus and Grafana: tokens per second, queue wait time and GPU memory per pod.

## CI for model releases
- GitHub Actions pipeline that runs regression tests and a small evaluation suite before any new model version is deployed; a drop of more than 2 points blocks the release.

## LLM cost and usage tracking
- Built a gateway in front of three model providers that logs prompt and completion tokens per team, adds request IDs and enforces monthly budgets.
- A weekly report shows cost per feature; moving a classification feature from a large model to a fine-tuned small model cut its cost by 85%.
- Added semantic caching for repeated questions, with a similarity threshold tuned to keep wrong cache hits below 0.5%.

## Load testing
- Used Locust to simulate 2,000 concurrent chat users and found that long prompts, not concurrency, caused most timeouts.

## Skills
- Python, Go; vLLM, Triton Inference Server, Docker, Kubernetes, Terraform.
