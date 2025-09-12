# ---------- Stage 1: build environment ----------
FROM python:3.11-slim AS builder

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates unzip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy and install dependencies 
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt \
    && pip install --no-cache-dir --prefix=/install awscli

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim

# system deps for runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy only installed packages from builder
COPY --from=builder /install /usr/local

# copy your app code
COPY rag_server.py entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh

# env vars 
ENV S3_BUCKET=rag-medical-us-east-1-209479269803 \
    MODEL_KEY=artifacts/model.gguf \
    CHROMA_TAR_KEY=artifacts/chroma_db_v_export.tar.gz \
    PYTHONUNBUFFERED=1

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
