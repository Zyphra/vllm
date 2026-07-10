#!/usr/bin/env bash
#SBATCH --job-name=tidar-tput
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=00:45:00
#SBATCH --output=/shared/home/jinzhao/tfscope/slurm-tidar-tput-%j.log

set -euo pipefail

REPO=${REPO:-/shared/home/jinzhao/tfscope/vllm-smoe-amd-bf16-v024}
CKPT=${CKPT:-/shared/home/henry/checkpoints/hf/smoediffusion_128k_64node/iter_0012600}
DATA=${DATA:-benchmarks/tidar/aime25_zpo_texts.json}
IMAGE=${IMAGE:-zyphra/rocm-primus:aiter_pa_swa}
MAX_TOKENS=${MAX_TOKENS:-5000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-12000}
N_SAMPLE_AR=${N_SAMPLE_AR:-1}
N_SAMPLE_TF=${N_SAMPLE_TF:-1}
RUN_AR=${RUN_AR:-1}
RUN_TF=${RUN_TF:-1}
LOGROOT=${LOGROOT:-/shared/home/jinzhao/tfscope/amd_iter12600_forcebos_mt${MAX_TOKENS}_n1_slurm_${SLURM_JOB_ID}}
CONTAINER=tidar-tput-${SLURM_JOB_ID}

mkdir -p "$LOGROOT"

cleanup() {
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d --name "$CONTAINER" --ipc=host --network=host \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v /shared:/shared -v "$REPO":/work -w /work \
    "$IMAGE" sleep infinity >/dev/null

docker exec "$CONTAINER" bash -lc \
    'pip install -q "setuptools>=77" "setuptools_scm>=8" wheel'
docker exec "$CONTAINER" bash -lc \
    'git config --global --add safe.directory /work && \
     SETUPTOOLS_SCM_PRETEND_VERSION=0.16.1 \
     SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.16.1 \
     pip install -q --no-build-isolation -e /work'

for pair in "1 1" "8 2" "16 3" "64 4"; do
    set -- $pair
    batch=$1
    gpu=$2
    mkdir -p "$LOGROOT/b$batch"
    docker exec \
        -e VLLM_CACHE_ROOT="/root/.cache/vllm-mt${MAX_TOKENS}-b$batch" \
        -e VLLM_VERSION_OVERRIDE=0.16.1 \
        "$CONTAINER" bash -lc \
        "cd /work && env CKPT=$CKPT DATA=$DATA BACKEND=ROCM_AITER_FA \
         GPU=$gpu BATCHES=$batch MAX_TOKENS=$MAX_TOKENS \
         MAX_MODEL_LEN=$MAX_MODEL_LEN N_SAMPLE_AR=$N_SAMPLE_AR \
         N_SAMPLE_TF=$N_SAMPLE_TF RUN_AR=$RUN_AR RUN_TF=$RUN_TF \
         LOGROOT=$LOGROOT/b$batch \
         bash benchmarks/tidar/run_iter12600_tput.sh" \
        > "$LOGROOT/b$batch.driver.log" 2>&1 &
done

wait
grep -H -E 'PATCH_PROBE_(CONTEXT|BEST)' "$LOGROOT"/b*/*.log \
    | tee "$LOGROOT/results.txt"
