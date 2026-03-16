#!/usr/bin/env bash
# Wait for both the tokenizer and the full pre-training corpus to be ready,
# then launch Phase 0 pre-training (500M config) on CUDA.
#
# Prerequisites:
#   - scripts/wait_and_tokenize.sh has completed → tokenizer_output/ exists
#   - data/pretrain/train.jsonl is complete (manifest.json has "done": true)
#
# Usage:
#   bash scripts/wait_and_pretrain.sh
#
# The training loop logs to logs/pretrain_500m.log and saves checkpoints
# to checkpoints/500m/.  It can be safely interrupted and resumed — the
# loop reads the latest checkpoint on startup.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/.local/bin/env 2>/dev/null || true

TOKENIZER_OUT=tokenizer_output
TRAIN_JSONL=data/pretrain/train.jsonl
MANIFEST=data/pretrain/manifest.json
CHECKPOINT_DIR=checkpoints/500m
LOG_DIR=logs
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

echo "[wait_and_pretrain] Waiting for tokenizer at ${TOKENIZER_OUT}/ ..."
while [ ! -f "${TOKENIZER_OUT}/tokenizer.json" ]; do
    echo "[wait_and_pretrain] $(date '+%H:%M:%S')  tokenizer not ready yet..."
    sleep 60
done
echo "[wait_and_pretrain] Tokenizer ready."

echo "[wait_and_pretrain] Waiting for data manifest to show done=true ..."
while true; do
    if [ -f "$MANIFEST" ]; then
        DONE=$(python3 -c "import json; d=json.load(open('$MANIFEST')); print(d.get('done', False))")
        if [ "$DONE" = "True" ]; then
            echo "[wait_and_pretrain] Data pipeline complete."
            break
        fi
        TOKENS=$(python3 -c "import json; d=json.load(open('$MANIFEST')); print(d.get('total_tokens_estimate', 0))")
        echo "[wait_and_pretrain] $(date '+%H:%M:%S')  tokens so far: $TOKENS"
    fi
    sleep 120
done

echo "[wait_and_pretrain] Launching Phase 0 pre-training (500M, 10B tokens)..."

uv run srm-pretrain \
    --config 500m \
    --backend cuda \
    --data_path "$TRAIN_JSONL" \
    --tokenizer_path "$TOKENIZER_OUT" \
    --output_dir "$CHECKPOINT_DIR" \
    --max_tokens 10_000_000_000 \
    2>&1 | tee "$LOG_DIR/pretrain_500m.log"

echo "[wait_and_pretrain] Pre-training complete. Checkpoints in: $CHECKPOINT_DIR"
