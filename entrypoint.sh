#!/usr/bin/env bash
set -euo pipefail

# Expected env: S3_BUCKET, MODEL_KEY, CHROMA_TAR_KEY
mkdir -p /opt/data/models /opt/data

if [ ! -f /opt/data/models/model.gguf ]; then
  echo "➡️  downloading model from s3://$S3_BUCKET/$MODEL_KEY"
  aws s3 cp "s3://$S3_BUCKET/$MODEL_KEY" /opt/data/models/model.gguf
fi

if [ ! -d /opt/data/chroma_db_v_export ]; then
  echo "➡️  downloading chroma export from s3://$S3_BUCKET/$CHROMA_TAR_KEY"
  aws s3 cp "s3://$S3_BUCKET/$CHROMA_TAR_KEY" /tmp/chroma.tar.gz
  tar -xzf /tmp/chroma.tar.gz -C /opt/data
fi

# Run API
exec python -m uvicorn rag_server:app --host 0.0.0.0 --port 8000
