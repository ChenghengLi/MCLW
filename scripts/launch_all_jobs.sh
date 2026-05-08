#!/usr/bin/env bash
# Master launcher: submits 9 runai H100 jobs in parallel
#   3 models × 3 experiments (2 head-to-head, 3 translation curve, 4 k=2 soft_cycle).
# All H100 jobs are preemptible per the cluster policy.
#
# Run on the bastion / submission host as user `lichen`:
#   bash /home/lichen/MCLW/scripts/launch_all_jobs.sh
set -euo pipefail

PROJ=dlab-lichen
IMAGE=pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
NODE_POOL=h100

# Pip deps for inside the container (matches MCLW/CLAUDE.md and excludes vllm).
PIP_DEPS="transformers accelerate tqdm wandb python-dotenv huggingface_hub \
  numpy scipy matplotlib seaborn pandas pyyaml scikit-learn plotly kaleido \
  nltk python-levenshtein sentence-transformers datasets bitsandbytes"

# (short_name, hf_model_id, int8_flag)
declare -a JOBS=(
  "llama8b   meta-llama/Llama-3.1-8B-Instruct        "
  "qwen7b    Qwen/Qwen2.5-7B-Instruct                "
  "mistral7b mistralai/Mistral-7B-Instruct-v0.3      "
)

EXPS=(2 3 4)

submit_job () {
  local exp="$1"
  local short="$2"
  local model="$3"
  local int8="$4"
  local name="mclw-e${exp}-${short}"
  local wrapper="/home/lichen/MCLW/scripts/run_exp${exp}.sh"

  echo "[launch] submitting $name  (exp=$exp model=$model int8='${int8}')"

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
    --command -- bash -c "groupadd -g 30204 lichengrp 2>/dev/null || true && useradd -u 316680 -g 30204 -d /home/lichen -s /bin/bash -M lichen 2>/dev/null || true && cat /home/lichen/setup.sh | bash && pip install --timeout 120 --retries 5 ${PIP_DEPS} && su lichen -s /bin/bash ${wrapper} ${model} ${int8}"
}

for row in "${JOBS[@]}"; do
  read -r short model int8 <<< "$row"
  for E in "${EXPS[@]}"; do
    submit_job "$E" "$short" "$model" "$int8"
  done
done

echo
echo "[launch] all 9 jobs submitted. Monitor with:"
echo "  runai list jobs -p $PROJ | grep mclw-"
echo "  runai logs <job-name> -p $PROJ"
