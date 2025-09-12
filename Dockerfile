FROM python:3.11-slim

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir awscli  # <— add this

COPY rag_server.py entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh

# default env vars (override in App Runner)
ENV S3_BUCKET=rag-medical-us-east-1-209479269803 \
    MODEL_KEY=artifacts/model.gguf \
    CHROMA_TAR_KEY=artifacts/chroma_db_v_export.tar.gz \
    PYTHONUNBUFFERED=1

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]

