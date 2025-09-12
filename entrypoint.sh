#!/usr/bin/env bash
set -euo pipefail

# Expected env: S3_BUCKET, MODEL_KEY, CHROMA_TAR_KEY
# Local target paths:
MODEL_DST="/opt/data/models/model.gguf"
CHROMA_DIR_DST="/opt/data/chroma_db_v_export"

mkdir -p /opt/data/models /opt/data

# --- Download model ---
if [ ! -f "$MODEL_DST" ]; then
  if [ -n "${S3_BUCKET:-}" ] && [ -n "${MODEL_KEY:-}" ]; then
    echo "➡️  downloading model from s3://$S3_BUCKET/$MODEL_KEY"
    aws s3 cp "s3://$S3_BUCKET/$MODEL_KEY" "$MODEL_DST"
  else
    echo "⚠️  MODEL missing and no S3 env provided; starting anyway (server may fail if MODEL_PATH is required)" >&2
  fi
fi

# --- Download Chroma DB export ---
if [ ! -d "$CHROMA_DIR_DST" ]; then
  if [ -n "${S3_BUCKET:-}" ] && [ -n "${CHROMA_TAR_KEY:-}" ]; then
    echo "➡️  downloading chroma export from s3://$S3_BUCKET/$CHROMA_TAR_KEY"
    aws s3 cp "s3://$S3_BUCKET/$CHROMA_TAR_KEY" /tmp/chroma.tar.gz
    tar -xzf /tmp/chroma.tar.gz -C /opt/data
  else
    echo "ℹ️  No Chroma archive configured; continuing without it."
  fi
fi

# --- Launch API ---
exec python -m uvicorn rag_server:app --host 0.0.0.0 --port 8000
