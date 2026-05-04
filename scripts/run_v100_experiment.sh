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
# Notes for the EPFL RCP / pytorch:2.6.0-cuda image:
#   - python lives at /opt/conda/bin/python; the system PATH for su'd users
#     does not include it, so we hardcode the path.
#   - We deliberately do NOT use `set -u` or `set -o pipefail` here because
#     stderr from those shell errors gets lost when the pod is preempted.
#     Plain `set -e` plus explicit `|| { echo ... ; exit 1; }` checks make
#     diagnosis under preemption much easier.
#   - python is invoked with -u so each line is flushed immediately to the
#     pod log (otherwise CPython buffers when stdout is a pipe through tee).
set -e

MODEL="${1:?usage: run_v100_experiment.sh <hf_model_id> [domains...]}"
shift
if [ "$#" -eq 0 ]; then
  DOMAINS=(wiki news social abstract)
else
  DOMAINS=("$@")
fi

PY=/opt/conda/bin/python
if [ ! -x "$PY" ]; then
  PY="$(command -v python3 || command -v python || true)"
fi
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  echo "ERROR: no python interpreter found (tried /opt/conda/bin/python, python3, python)" >&2
  exit 1
fi

REPO_DIR=/home/lichen/MCLW_runai
cd "$REPO_DIR" || { echo "ERROR: cannot cd into $REPO_DIR" >&2; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$REPO_DIR/runs/$STAMP"
mkdir -p "$RUN_DIR"

# Heartbeat file we can check from another job if pod gets preempted.
HEARTBEAT="$RUN_DIR/HEARTBEAT"
echo "started_at=$(date -Iseconds)" > "$HEARTBEAT"
echo "model=$MODEL" >> "$HEARTBEAT"
echo "domains=${DOMAINS[*]}" >> "$HEARTBEAT"
echo "git_rev=$(git rev-parse HEAD)" >> "$HEARTBEAT"
echo "host=$(hostname)" >> "$HEARTBEAT"
echo "python=$PY ($($PY --version 2>&1))" >> "$HEARTBEAT"

cat > "$RUN_DIR/MANIFEST" <<EOF
stamp:    $STAMP
model:    $MODEL
git_rev:  $(git rev-parse HEAD)
domains:  ${DOMAINS[*]}
hostname: $(hostname)
python:   $PY ($($PY --version 2>&1))
nvidia:   $(nvidia-smi -L 2>/dev/null | head -1 || echo "no GPU")
EOF

echo "[$(date -Iseconds)] starting run; manifest:"
cat "$RUN_DIR/MANIFEST"
echo

export HF_HOME=/home/lichen/hf_cache
mkdir -p "$HF_HOME"

for D in "${DOMAINS[@]}"; do
  echo
  echo "============================================================"
  echo "[$(date -Iseconds)] DOMAIN: $D"
  echo "============================================================"
  echo "domain=$D started_at=$(date -Iseconds)" >> "$HEARTBEAT"
  set -o pipefail
  "$PY" -u scripts/generate_curated_dataset.py \
    --domain "$D" \
    --states 7 --overlaps 0 \
    --max-tokens 512 \
    --decoding greedy \
    --batch-size "${BATCH_SIZE:-8}" \
    --model "$MODEL" 2>&1 | tee "$RUN_DIR/${D}.log"
  rc=$?
  set +o pipefail
  if [ "$rc" -ne 0 ]; then
    echo "domain=$D FAILED rc=$rc at=$(date -Iseconds)" >> "$HEARTBEAT"
    echo "ERROR: python exited $rc on domain $D — see $RUN_DIR/${D}.log" >&2
    exit "$rc"
  fi
  echo "domain=$D finished_at=$(date -Iseconds)" >> "$HEARTBEAT"
done

echo "finished_at=$(date -Iseconds)" >> "$HEARTBEAT"

echo
echo "============================================================"
echo "[$(date -Iseconds)] DONE."
ls -la data/ | tail -10
echo "[$(date -Iseconds)] Run dir: $RUN_DIR"
