# Public demo image (DEMO_MODE): fictional sample profile, no PyTorch, Chinese fonts for the PDF CV.
# Works on ModelScope 创空间 and Hugging Face Spaces (Docker type, port 7860). See DEPLOY.md.
FROM python:3.11-slim

# A mirror inside China (ModelScope builds there); for Hugging Face use --build-arg PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_INDEX_URL=${PIP_INDEX_URL} PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    DEMO_MODE=1 APP_THEME=cute HOME=/tmp

RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null; \
    apt-get update && apt-get install -y --no-install-recommends fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-demo.txt .
RUN pip install -r requirements-demo.txt
COPY . .
RUN chmod -R a+rwX /app

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
