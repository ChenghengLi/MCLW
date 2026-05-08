#!/usr/bin/env bash
# Launch the 9 method-split exp-2 cells: 3 methods × 3 models.
# Each cell does 4 domains × 100 prompts × n=200 + 3 attacks (clean / random-sub
# is implicit / ZH-back-trans / DIPPER), restricted to ONE watermarking method.
# This 3× parallelizes the longest pole vs. the monolithic exp-2 job.
#
# Run on the bastion as user lichen:
#   bash /home/lichen/MCLW/scripts/launch_exp2_split.sh
set -euo pipefail

PROJ=dlab-lichen
IMAGE=pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
NODE_POOL=h100

PIP_DEPS="transformers accelerate tqdm wandb python-dotenv huggingface_hub \
  numpy scipy matplotlib seaborn pandas pyyaml scikit-learn plotly kaleido \
  nltk python-levenshtein sentence-transformers datasets bitsandbytes"

declare -a MODELS=(
  "llama8b   meta-llama/Llama-3.1-8B-Instruct"
  "qwen7b    Qwen/Qwen2.5-7B-Instruct"
  "mistral7b mistralai/Mistral-7B-Instruct-v0.3"
)

METHODS=(mcl kgw sweet)

submit_one () {
  local short="$1"
  local model="$2"
  local method="$3"
  local name="mclw-e2v2-${method}-${short}"
  local wrapper="/home/lichen/MCLW/scripts/run_exp2_v2.sh"
  echo "[launch] submitting $name (method=$method model=$model)"
  runai submit \
    --name "$name" \
    --interactive \
    -p "$PROJ" \
    -i "$IMAGE" \
    -g 1 \
    --node-pools "$NODE_POOL" \
    --large-shm \
    --preemptible \
    --existing-pvc claimname=home,path=/home/lichen \
    --command -- bash -c "groupadd -g 30204 lichengrp 2>/dev/null || true && useradd -u 316680 -g 30204 -d /home/lichen -s /bin/bash -M lichen 2>/dev/null || true && cat /home/lichen/setup.sh | bash && pip install --timeout 120 --retries 5 ${PIP_DEPS} && su lichen -s /bin/bash ${wrapper} ${model} ${method}"
}

for row in "${MODELS[@]}"; do
  read -r short model <<< "$row"
  for m in "${METHODS[@]}"; do
    submit_one "$short" "$model" "$m"
  done
done

echo
echo "[launch] all 9 method-split exp-2 jobs submitted"
echo "  combined with the existing 3 exp-3 + 3 exp-4 jobs running, total = 15 parallel jobs."
echo "  monitor with: runai list jobs -p $PROJ | grep mclw-"
