#!/usr/bin/env bash
# Wait for the tokenizer, corpus, and pre-tokenized binary data to be ready,
# then launch Phase 0 pre-training (500M config) on CUDA.
#
# Prerequisites:
#   - scripts/wait_and_tokenize.sh has completed → tokenizer_output/ exists
#   - data/pretrain/train.jsonl is complete (manifest.json has "done": true)
#
# This script also handles the pre-tokenization step (JSONL → binary .bin files)
# which eliminates the HuggingFace tokenizers Rust/rayon/futex deadlock during
# training.  Pre-tokenization is idempotent — if data/pretrain_tokenized/train.bin
# already exists it is skipped.
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
TOKENIZED_DIR=data/pretrain_tokenized
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

# Pre-tokenize the JSONL corpus into flat uint16 binary files.
# This step eliminates HuggingFace tokenizers Rust thread pool usage during
# training, which causes a futex deadlock when combined with PyTorch's CUDA
# background threads (observed as 98% GPU + 100% CPU with zero logged steps).
if [ -f "${TOKENIZED_DIR}/train.bin" ]; then
    echo "[wait_and_pretrain] Pre-tokenized data already exists at ${TOKENIZED_DIR}/, skipping."
else
    echo "[wait_and_pretrain] Pre-tokenizing corpus → ${TOKENIZED_DIR}/ ..."
    TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 uv run python data/tokenize_dataset.py \
        --input "$TRAIN_JSONL" \
        --tokenizer "$TOKENIZER_OUT" \
        --output_dir "$TOKENIZED_DIR" \
        2>&1 | tee "$LOG_DIR/tokenize.log"
    echo "[wait_and_pretrain] Pre-tokenization complete."
fi

echo "[wait_and_pretrain] Launching Phase 0 pre-training (500M, 10B tokens)..."

# PYTHONUNBUFFERED=1 forces Python to flush stdout after every write rather than
# buffering into 8 KB blocks.  Without it, output piped through tee is block-
# buffered: the startup lines sit in the pipe buffer and the log file stays at 0
# bytes until the first flush=True print fires (at step 0, ~100 s in).
# With it, every print() lands in the log immediately.
PYTHONUNBUFFERED=1 uv run srm-pretrain \
    --config 500m \
    --backend cuda \
    --data_path "$TOKENIZED_DIR" \
    --output_dir "$CHECKPOINT_DIR" \
    --max_tokens 10_000_000_000 \
    2>&1 | tee "$LOG_DIR/pretrain_500m.log"

echo "[wait_and_pretrain] Pre-training complete. Checkpoints in: $CHECKPOINT_DIR"
