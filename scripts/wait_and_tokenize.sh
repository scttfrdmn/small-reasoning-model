#!/usr/bin/env bash
# Wait until data/pretrain/train.jsonl has at least TARGET_GB of data,
# then train the tokenizer on it.  Intended to run in a tmux session
# concurrently with the data download so tokenizer training starts
# automatically as soon as there is enough corpus text.
#
# Usage:
#   bash scripts/wait_and_tokenize.sh
#
# Adjust TARGET_GB to control how much data to collect before training.
# 2 GB of text (~500M tokens at 4 chars/token) is well above the minimum
# needed for BPE to learn 32768 merges on a diverse English + math corpus.
set -euo pipefail
cd "$(dirname "$0")/.."   # always run from repo root
source ~/.local/bin/env 2>/dev/null || true

TARGET_GB=2
TRAIN_JSONL=data/pretrain/train.jsonl
TOKENIZER_OUT=tokenizer_output
LOG_DIR=logs
mkdir -p "$LOG_DIR"

echo "[wait_and_tokenize] Waiting for ${TRAIN_JSONL} to reach ${TARGET_GB} GB..."

while true; do
    if [ -f "$TRAIN_JSONL" ]; then
        SIZE_BYTES=$(stat --format="%s" "$TRAIN_JSONL" 2>/dev/null || echo 0)
        SIZE_GB=$(python3 -c "print(f'{$SIZE_BYTES / 1024**3:.2f}')")
        TARGET_BYTES=$(python3 -c "print(int($TARGET_GB * 1024**3))")
        echo "[wait_and_tokenize] $(date '+%H:%M:%S')  ${SIZE_GB} GB / ${TARGET_GB} GB"
        if [ "$SIZE_BYTES" -ge "$TARGET_BYTES" ]; then
            echo "[wait_and_tokenize] Target reached. Starting tokenizer training..."
            break
        fi
    else
        echo "[wait_and_tokenize] $(date '+%H:%M:%S')  Waiting for ${TRAIN_JSONL} to appear..."
    fi
    sleep 60
done

uv run srm-tokenizer \
    --mode corpus \
    --data "$TRAIN_JSONL" \
    --output "$TOKENIZER_OUT" \
    2>&1 | tee "$LOG_DIR/tokenizer.log"

echo "[wait_and_tokenize] Tokenizer training complete. Output: $TOKENIZER_OUT"
