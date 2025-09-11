#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:=/models/model.gguf}"
: "${MODEL_URL:=}"

mkdir -p "$(dirname "$MODEL_PATH")"

if [ -f "$MODEL_PATH" ]; then
  echo "Model already present at $MODEL_PATH"
else
  if [ -z "$MODEL_URL" ]; then
    echo "ERROR: MODEL_URL not set and model missing: $MODEL_PATH" >&2
    exit 1
  fi
  echo "Downloading model from $MODEL_URL -> $MODEL_PATH"
  if [[ "$MODEL_URL" == s3://* ]]; then
    aws s3 cp "$MODEL_URL" "$MODEL_PATH"
  else
    curl -L "$MODEL_URL" -o "$MODEL_PATH"
  fi
fi
