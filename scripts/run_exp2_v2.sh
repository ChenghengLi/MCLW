#!/usr/bin/env bash
# Independent runner: experiment 2 (head-to-head MCL vs KGW vs SWEET) on ONE model.
# Called inside the runai container by `su lichen -s /bin/bash`.
#
# Usage:  bash run_exp2.sh <hf_model_id> [--int8]
set -euo pipefail

MODEL="${1:?missing model id}"
METHOD="${2:-}"   # optional: mcl | kgw | sweet (split exp 2 (NLTK FIX V2) across method jobs)
INT8="${3:-}"

export PATH="/opt/conda/bin:${PATH:-}"
cd /home/lichen/MCLW

# Use container-local /tmp for HF cache to avoid blowing the home PVC quota.
# Each runai pod has ~1 TB of NVMe scratch on /tmp.
export HF_HOME=/tmp/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/home/lichen/MCLW/src:/home/lichen/MCLW/scripts:${PYTHONPATH:-}"

if [[ -f /home/lichen/.env ]]; then
  set -a; . /home/lichen/.env; set +a
fi

# Pre-fetch NLTK resources DIPPER needs (punkt_tab + punkt). Cached under home.
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" || true

OUT_ROOT=/home/lichen/MCLW/data/v7_min_nltkfix
mkdir -p "$OUT_ROOT"

METHOD_FLAG=""
[[ -n "$METHOD" ]] && METHOD_FLAG="--only-method $METHOD"

echo "[exp2] model=$MODEL method=${METHOD:-all} int8=$INT8 user=$(whoami) cwd=$(pwd)"
python3 /home/lichen/MCLW/scripts/exp_min_runner.py \
  --exp 2 --model "$MODEL" \
  --max-tokens 200 --n-prompts 100 \
  --out-root "$OUT_ROOT" \
  $METHOD_FLAG \
  $INT8
