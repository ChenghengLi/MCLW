#!/usr/bin/env bash
# Independent runner: experiment 3 (MCL-only translation curve, 25-prompt subsample)
# on ONE model.
#
# Usage:  bash run_exp3.sh <hf_model_id> [--int8]
set -euo pipefail

MODEL="${1:?missing model id}"
INT8="${2:-}"

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

echo "[exp3] model=$MODEL int8=$INT8 user=$(whoami) cwd=$(pwd)"
python3 /home/lichen/MCLW/scripts/exp_min_runner.py \
  --exp 3 --model "$MODEL" \
  --max-tokens 200 --n-prompts 25 \
  --out-root "$OUT_ROOT" \
  $INT8
