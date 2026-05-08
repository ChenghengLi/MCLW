#!/usr/bin/env bash
# Launch the calibration S-sweep: 5 S-values × 3 models = 15 independent jobs.
# Each cell does 4 domains × 25 prompts × n=200, watermarked + non-watermarked.
# This empirically anchors Theorem 1's closed-form S*(n, ρ, α) prediction.
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

S_VALUES=(2 3 5 7 11)

submit_one () {
  local short="$1"
  local model="$2"
  local s="$3"
  local name="mclw-e5-s${s}-${short}"
  local wrapper="/home/lichen/MCLW/scripts/run_exp5.sh"
  echo "[launch] submitting $name (S=$s model=$model)"
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
    --command -- bash -c "groupadd -g 30204 lichengrp 2>/dev/null || true && useradd -u 316680 -g 30204 -d /home/lichen -s /bin/bash -M lichen 2>/dev/null || true && cat /home/lichen/setup.sh | bash && pip install --timeout 120 --retries 5 ${PIP_DEPS} && su lichen -s /bin/bash ${wrapper} ${model} ${s}"
}

for row in "${MODELS[@]}"; do
  read -r short model <<< "$row"
  for s in "${S_VALUES[@]}"; do
    submit_one "$short" "$model" "$s"
  done
done

echo
echo "[launch] all 15 calibration-sweep jobs submitted"
echo "  monitor with: runai list jobs -p $PROJ | grep mclw-e5-"
