#!/usr/bin/env bash
#SBATCH --job-name=tidar-lockstep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00
#SBATCH --output=/shared/home/jinzhao/tfscope/slurm-tidar-lockstep-%j.log

set -euo pipefail

REPO=${REPO:-/shared/home/jinzhao/tfscope/vllm-smoe-amd-bf16-v024}
CKPT=${CKPT:-/shared/home/henry/checkpoints/hf/smoediffusion_128k_64node/iter_0012600}
DATA=${DATA:-benchmarks/tidar/aime25_zpo_texts.json}
IMAGE=${IMAGE:-zyphra/rocm-primus:aiter_pa_swa}
LOGROOT=${LOGROOT:-/shared/home/jinzhao/tfscope/amd_lockstep_steady_${SLURM_JOB_ID}}
CONTAINER=tidar-lockstep-${SLURM_JOB_ID}
PYTHON=python
BATCHES=${BATCHES:-"1 8 16 64"}
RUN_AR=${RUN_AR:-1}
RUN_TF=${RUN_TF:-1}
PROFILE=${PROFILE:-0}
START_GPU=${START_GPU:-1}

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
    'pip install -q "setuptools>=77" "setuptools_scm>=8" wheel && \
     git config --global --add safe.directory /work && \
     SETUPTOOLS_SCM_PRETEND_VERSION=0.24.0 \
     SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.24.0 \
     pip install -q --no-build-isolation -e /work'

common_env=(
    -e VLLM_USE_V2_MODEL_RUNNER=1
    -e VLLM_ENABLE_V1_MULTIPROCESSING=0
    -e VLLM_ATTENTION_BACKEND=ROCM_AITER_FA
    -e VLLM_SKIP_SDPA_PREINIT=1
    -e VLLM_CCA_TRITON=1
    -e VLLM_CCA_TRITON_FUSION_ENABLED=0
    -e VLLM_CCA_AMD_CONV_UNFOLD=0
    -e VLLM_TIDAR_ROUTER_PAD=1
    -e VLLM_ROCM_USE_AITER=1
    -e VLLM_ROCM_USE_AITER_MHA=1
    -e VLLM_ROCM_USE_AITER_MOE=1
    -e VLLM_ROCM_MOE_PADDING=1
    -e VLLM_TIDAR_SMOE_MOE_OP=0
    -e PATCH_PROBE_STEADY=1
    -e PATCH_PROBE_EXACT_CUDAGRAPH_BATCH=1
)
if [[ "$PROFILE" == "1" ]]; then
    common_env+=(
        -e PATCH_PROBE_PROFILE=1
        -e PATCH_PROBE_PROFILE_MAX_EVENTS=128
    )
fi

run_ar() {
    local batch=$1
    local gpu=$2
    docker exec "${common_env[@]}" \
        -e HIP_VISIBLE_DEVICES="$gpu" -e CUDA_VISIBLE_DEVICES="$gpu" \
        -e PATCH_PROBE_STEADY_BATCH="$batch" \
        "$CONTAINER" bash -lc \
        "cd /work && $PYTHON -u benchmarks/tidar/probe_v2_ar.py \
         --ckpt '$CKPT' --dataset '$DATA' --batch '$batch' --num-prompts 1 \
         --max-num-seqs '$batch' --n-sample '$batch' --max-tokens 512 \
         --warmup-tokens 64 --repeats 1 --seed 0 --target-temp 0.6 \
         --max-model-len 12000 --max-num-batched-tokens 8192 \
         --gpu-memory-utilization 0.65 --backend ROCM_AITER_FA \
         --cudagraph-mode FULL_AND_PIECEWISE --prompt-token-ids --force-bos \
         --ignore-eos" >"$LOGROOT/ar_b${batch}.log" 2>&1
}

run_tf() {
    local batch=$1
    local gpu=$2
    docker exec "${common_env[@]}" \
        -e HIP_VISIBLE_DEVICES="$gpu" -e CUDA_VISIBLE_DEVICES="$gpu" \
        -e PATCH_PROBE_STEADY_BATCH="$batch" \
        -e VLLM_TIDAR_TWO_FORWARD=1 \
        -e VLLM_TIDAR_USE_TF_PAGED_ATTENTION=1 \
        -e VLLM_TIDAR_TF_PAGED_NO_SPLITS=1 \
        -e VLLM_TIDAR_FA_NO_SPLITS=1 \
        "$CONTAINER" bash -lc \
        "cd /work && $PYTHON -u benchmarks/tidar/probe_v2_tidar_nv.py \
         --ckpt '$CKPT' --dataset '$DATA' --batch '$batch' --num-prompts 1 \
         --max-num-seqs '$batch' --n-sample '$batch' --max-tokens 512 \
         --warmup-tokens 64 --repeats 1 --seed 0 --target-temp 0.6 \
         --draft-temp 0 --num-spec-tokens 16 --max-model-len 12000 \
         --max-num-batched-tokens 8192 --gpu-memory-utilization 0.65 \
         --backend ROCM_AITER_FA --cudagraph-mode FULL_AND_PIECEWISE \
         --prompt-token-ids --force-bos --ignore-eos" \
        >"$LOGROOT/tf_b${batch}.log" 2>&1
}

pids=()
gpu=$START_GPU
for batch in $BATCHES; do
    if [[ "$RUN_AR" == "1" ]]; then
        run_ar "$batch" "$gpu" &
        pids+=("$!")
        gpu=$((gpu + 1))
    fi
    if [[ "$RUN_TF" == "1" ]]; then
        run_tf "$batch" "$gpu" &
        pids+=("$!")
        gpu=$((gpu + 1))
    fi
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

grep -H -E 'PATCH_PROBE_(CONTEXT|BEST)' "$LOGROOT"/*.log \
    >"$LOGROOT/results.txt" || true
cat "$LOGROOT/results.txt"
exit "$status"
