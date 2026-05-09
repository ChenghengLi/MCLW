#!/usr/bin/env bash
# Backfill the Mistral×code Exp.~3 cell that didn't complete in the original
# cluster window (paper §08, §14 caveat). Appends 25 prompts × 4 NLLB pivots
# to the existing data/v7_min/exp3_mistral-7b-instruct-v0-3/records.jsonl.
set -euo pipefail

export PATH="/opt/conda/bin:${PATH:-}"
cd /home/lichen/MCLW

export HF_HOME=/tmp/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/home/lichen/MCLW/src:/home/lichen/MCLW/scripts:${PYTHONPATH:-}"

if [[ -f /home/lichen/.env ]]; then
  set -a; . /home/lichen/.env; set +a
fi

python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" || true

OUT_ROOT=/home/lichen/MCLW/data/v7_min
mkdir -p "$OUT_ROOT"

echo "[exp3-mistral-code] backfilling missing cell, user=$(whoami) cwd=$(pwd)"
python3 /home/lichen/MCLW/scripts/exp_min_runner.py \
  --exp 3 --model mistralai/Mistral-7B-Instruct-v0.3 \
  --max-tokens 200 --n-prompts 25 \
  --only-domain code \
  --append \
  --out-root "$OUT_ROOT"

echo "[exp3-mistral-code] DONE"
