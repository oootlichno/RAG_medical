#!/bin/sh
set -eu

# Expected env: S3_BUCKET, MODEL_KEY, CHROMA_TAR_KEY (when using S3)
# Files we want on disk:
MODEL_DST="/opt/data/models/model.gguf"
CHROMA_DIR_DST="/opt/data/chroma_db_v_export"

mkdir -p /opt/data/models /opt/data

need_aws=0
# Download model if missing and env points to S3
if [ ! -f "$MODEL_DST" ]; then
  if [ -n "${S3_BUCKET:-}" ] && [ -n "${MODEL_KEY:-}" ]; then
    echo "➡️  downloading model from s3://$S3_BUCKET/$MODEL_KEY"
    aws s3 cp "s3://$S3_BUCKET/$MODEL_KEY" "$MODEL_DST"
    need_aws=1
  else
    echo "⚠️  MODEL missing and no S3 env provided; starting anyway (server may fail if MODEL_PATH is required)" >&2
  fi
fi

# Download chroma export if missing and env points to S3
if [ ! -d "$CHROMA_DIR_DST" ]; then
  if [ -n "${S3_BUCKET:-}" ] && [ -n "${CHROMA_TAR_KEY:-}" ]; then
    echo "➡️  downloading chroma export from s3://$S3_BUCKET/$CHROMA_TAR_KEY"
    aws s3 cp "s3://$S3_BUCKET/$CHROMA_TAR_KEY" /tmp/chroma.tar.gz
    tar -xzf /tmp/chroma.tar.gz -C /opt/data
    need_aws=1
  else
    echo "ℹ️  No Chroma archive configured; continuing without it."
  fi
fi

# Run API (defaults in rag_server.py expect these paths)
exec python -m uvicorn rag_server:app --host 0.0.0.0 --port 8000

