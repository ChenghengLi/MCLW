#!/usr/bin/env bash
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
[[ -f /home/lichen/.env ]] && set -a && source /home/lichen/.env && set +a
echo "[fpr] model=$MODEL int8=$INT8"
python3 /home/lichen/MCLW/scripts/measure_fpr.py --model "$MODEL" --n-prompts 50 --max-tokens 200 --seed "${SEED:-42}" $INT8
