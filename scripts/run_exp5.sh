#!/usr/bin/env bash
# Calibration S-sweep cell. Runs MCL at one (model, S) on
# 4 domains × 25 prompts × n=200 + non-watermarked baseline.
# Records both watermarked and non-watermarked phi/z so empirical FPR can
# be plotted against the closed-form Theorem 1 prediction.
#
# Usage (inside container as user lichen):
#   bash run_exp5.sh <hf_model_id> <S>  [--int8]
set -euo pipefail

MODEL="${1:?missing model id}"
S_VAL="${2:?missing S value}"
INT8="${3:-}"

export PATH="/opt/conda/bin:${PATH:-}"
cd /home/lichen/MCLW
export PYTHONPATH="/home/lichen/MCLW/src:/home/lichen/MCLW/scripts:${PYTHONPATH:-}"

OUT_ROOT="/home/lichen/MCLW/data/v7_min"
mkdir -p "$OUT_ROOT"

[[ -f /home/lichen/.env ]] && set -a && source /home/lichen/.env && set +a

python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" || true

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=warning
export TOKENIZERS_PARALLELISM=false

export HF_HOME=/tmp/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

echo "[exp5] model=$MODEL S=$S_VAL int8=$INT8 user=$(whoami) cwd=$(pwd)"
python3 /home/lichen/MCLW/scripts/exp_min_runner.py \
  --exp 5 --model "$MODEL" --S "$S_VAL" \
  --max-tokens 200 --n-prompts 25 \
  --out-root "$OUT_ROOT" \
  $INT8
