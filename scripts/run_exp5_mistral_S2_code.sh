#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/conda/bin:${PATH:-}"
cd /home/lichen/MCLW
export HF_HOME=/tmp/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/home/lichen/MCLW/src:/home/lichen/MCLW/scripts:${PYTHONPATH:-}"
[[ -f /home/lichen/.env ]] && { set -a; . /home/lichen/.env; set +a; }
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" || true
echo "[exp5-mistral-S2-code] backfilling Mistral×code at S=2"
python3 /home/lichen/MCLW/scripts/exp_min_runner.py \
  --exp 5 --S 2 --model mistralai/Mistral-7B-Instruct-v0.3 \
  --max-tokens 200 --n-prompts 25 \
  --only-domain code --append \
  --out-root /home/lichen/MCLW/data/v7_min
echo "[exp5-mistral-S2-code] DONE"
