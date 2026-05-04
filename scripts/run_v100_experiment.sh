#!/bin/bash
# Cross-domain MCL generation on a single V100 (or any single GPU).
#
# Usage (inside an EPFL RCP runai container that has /home/lichen mounted):
#     bash scripts/run_v100_experiment.sh <hf_model_id> [domain1 domain2 ...]
#
# Example:
#     bash scripts/run_v100_experiment.sh google/gemma-3-1b-it
#     bash scripts/run_v100_experiment.sh meta-llama/Llama-3.2-3B-Instruct
#
# For each domain, runs generate_curated_dataset.py with:
#     --states 7 --overlaps 0 --max-tokens 512 --decoding greedy
#
# Outputs:
#     data/curated_wiki_dataset_<stamp>/         (one per domain run)
#     runs/<stamp>/<domain>.log                  (stdout/stderr per domain)
#     runs/<stamp>/MANIFEST                      (model + git rev + stamp)
set -euo pipefail

MODEL="${1:?usage: run_v100_experiment.sh <hf_model_id> [domains...]}"
shift
DOMAINS=("${@:-wiki news social abstract}")

cd /home/lichen/MCLW_runai
echo "[$(date)] cwd: $(pwd)"
echo "[$(date)] git rev: $(git rev-parse HEAD)"
echo "[$(date)] model:    $MODEL"
echo "[$(date)] domains:  ${DOMAINS[*]}"
nvidia-smi -L || true

export HF_HOME=/home/lichen/hf_cache
mkdir -p "$HF_HOME"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="runs/$STAMP"
mkdir -p "$RUN_DIR"

cat > "$RUN_DIR/MANIFEST" <<EOF
stamp:    $STAMP
model:    $MODEL
git_rev:  $(git rev-parse HEAD)
domains:  ${DOMAINS[*]}
hostname: $(hostname)
nvidia:   $(nvidia-smi -L 2>/dev/null | head -1 || echo "no GPU")
EOF
echo "[$(date)] Manifest:"; cat "$RUN_DIR/MANIFEST"

for D in "${DOMAINS[@]}"; do
  echo
  echo "============================================================"
  echo "[$(date)] DOMAIN: $D"
  echo "============================================================"
  python scripts/generate_curated_dataset.py \
    --domain "$D" \
    --states 7 --overlaps 0 \
    --max-tokens 512 \
    --decoding greedy \
    --model "$MODEL" 2>&1 | tee "$RUN_DIR/${D}.log"
done

echo
echo "============================================================"
echo "[$(date)] DONE."
ls -la data/ | tail -10
echo "[$(date)] Run dir: $RUN_DIR"
