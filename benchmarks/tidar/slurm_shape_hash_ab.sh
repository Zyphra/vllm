#!/usr/bin/env bash
#SBATCH --job-name=tidar-shape-hash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00
#SBATCH --output=/shared/home/jinzhao/tfscope/slurm-tidar-shape-hash-%j.log

set -euo pipefail

REPO=${REPO:-/shared/home/jinzhao/tfscope/vllm-smoe-amd-bf16-v024}
CKPT=${CKPT:-/shared/home/henry/checkpoints/hf/smoediffusion_128k_64node/iter_0012600}
DATA=${DATA:-benchmarks/tidar/aime25_zpo_texts.json}
IMAGE=${IMAGE:-zyphra/rocm-primus:aiter_pa_swa}
LOGROOT=${LOGROOT:-/shared/home/jinzhao/tfscope/amd_shape_hash_ab_${SLURM_JOB_ID}}
CONTAINER=tidar-shape-hash-${SLURM_JOB_ID}
GPU=${GPU:-1}
ONLY=${ONLY:-all}

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

run_one() {
    local label=$1
    local batch=$2
    local aiter_moe=$3
    local cca_unfold=$4
    local backend=$5
    local layer_hash=${PATCH_PROBE_LAYER_HASH:-0}
    local trace_max_events=${PATCH_PROBE_TRACE_MAX_EVENTS:-8}
    local aiter_mha=0
    if [[ "$backend" == "ROCM_AITER_FA" ]]; then
        aiter_mha=1
    fi
    local log="$LOGROOT/${label}_b${batch}.log"

    docker exec \
        -e HIP_VISIBLE_DEVICES="$GPU" -e CUDA_VISIBLE_DEVICES="$GPU" \
        -e VLLM_USE_V2_MODEL_RUNNER=1 \
        -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        -e VLLM_ATTENTION_BACKEND="$backend" \
        -e VLLM_SKIP_SDPA_PREINIT=1 \
        -e VLLM_CCA_TRITON=1 \
        -e VLLM_CCA_TRITON_FUSION_ENABLED=0 \
        -e VLLM_CCA_AMD_CONV_UNFOLD="$cca_unfold" \
        -e VLLM_TIDAR_ROUTER_PAD=1 \
        -e VLLM_TIDAR_BATCH_INVARIANT_O_PROJ="${VLLM_TIDAR_BATCH_INVARIANT_O_PROJ:-0}" \
        -e VLLM_TIDAR_BATCH_INVARIANT_DENSE="${VLLM_TIDAR_BATCH_INVARIANT_DENSE:-0}" \
        -e VLLM_ROCM_USE_AITER=1 \
        -e VLLM_ROCM_USE_AITER_MHA="$aiter_mha" \
        -e VLLM_ROCM_USE_AITER_MOE="$aiter_moe" \
        -e VLLM_ROCM_USE_SKINNY_GEMM="${VLLM_ROCM_USE_SKINNY_GEMM:-1}" \
        -e VLLM_ROCM_MOE_PADDING=1 \
        -e VLLM_TIDAR_SMOE_MOE_OP=0 \
        -e VLLM_TIDAR_TWO_FORWARD=1 \
        -e VLLM_TIDAR_USE_TF_PAGED_ATTENTION=1 \
        -e VLLM_TIDAR_TF_PAGED_NO_SPLITS=1 \
        -e VLLM_TIDAR_FA_NO_SPLITS=1 \
        -e PATCH_PROBE_STEADY=1 \
        -e PATCH_PROBE_STEADY_BATCH="$batch" \
        -e PATCH_PROBE_TRACE=1 \
        -e PATCH_PROBE_TRACE_LABEL="${label}_b${batch}" \
        -e PATCH_PROBE_TRACE_MAX_EVENTS="$trace_max_events" \
        -e PATCH_PROBE_TRACE_MAX_REQS=1 \
        -e PATCH_PROBE_TRACE_TOPK=3 \
        -e PATCH_PROBE_LAYER_HASH="$layer_hash" \
        -e PATCH_PROBE_EXACT_CUDAGRAPH_BATCH=1 \
        "$CONTAINER" bash -lc \
        "cd /work && python -u benchmarks/tidar/probe_v2_tidar_nv.py \
         --ckpt '$CKPT' --dataset '$DATA' --batch '$batch' \
         --num-prompts 1 --max-num-seqs '$batch' --n-sample '$batch' \
         --max-tokens 128 \
         --warmup-tokens 64 --repeats 1 --seed 0 --target-temp 0.6 \
         --draft-temp 0 --num-spec-tokens 16 --max-model-len 12000 \
         --max-num-batched-tokens 8192 --gpu-memory-utilization 0.65 \
         --backend '$backend' --cudagraph-mode FULL_AND_PIECEWISE \
         --prompt-token-ids --force-bos --ignore-eos" >"$log" 2>&1
}

if [[ "$ONLY" == "all" || "$ONLY" == "core" ]]; then
    for batch in 1 64; do
        run_one baseline_aiter_moe "$batch" 1 0 ROCM_AITER_FA
    done
    for batch in 1 64; do
        run_one triton_moe "$batch" 0 0 ROCM_AITER_FA
    done
    for batch in 1 64; do
        run_one cca_unfold "$batch" 1 1 ROCM_AITER_FA
    done
fi
if [[ "$ONLY" == "all" || "$ONLY" == "triton_attention" \
        || "$ONLY" == "attention_layer" ]]; then
    for batch in 1 64; do
        run_one triton_attention "$batch" 1 0 TRITON_ATTN
    done
fi
if [[ "$ONLY" == "layer" || "$ONLY" == "attention_layer" \
        || "$ONLY" == "stage" || "$ONLY" == "cca_internal" ]]; then
    for batch in 1 64; do
        PATCH_PROBE_LAYER_HASH=1 PATCH_PROBE_TRACE_MAX_EVENTS=1 \
            run_one "${ONLY}_baseline" "$batch" 1 0 ROCM_AITER_FA
    done
fi
if [[ "$ONLY" == "no_skinny" ]]; then
    for batch in 1 64; do
        VLLM_ROCM_USE_SKINNY_GEMM=0 PATCH_PROBE_LAYER_HASH=1 \
            PATCH_PROBE_TRACE_MAX_EVENTS=1 \
            run_one no_skinny_baseline "$batch" 1 0 ROCM_AITER_FA
    done
fi
if [[ "$ONLY" == "no_skinny_triton_moe" ]]; then
    for batch in 1 64; do
        VLLM_ROCM_USE_SKINNY_GEMM=0 PATCH_PROBE_LAYER_HASH=1 \
            PATCH_PROBE_TRACE_MAX_EVENTS=1 \
            run_one no_skinny_triton_moe "$batch" 0 0 ROCM_AITER_FA
    done
fi
if [[ "$ONLY" == "invariant_o_proj" ]]; then
    for batch in 1 64; do
        VLLM_TIDAR_BATCH_INVARIANT_O_PROJ=1 PATCH_PROBE_LAYER_HASH=1 \
            PATCH_PROBE_TRACE_MAX_EVENTS=1 \
            run_one invariant_o_proj "$batch" 1 0 ROCM_AITER_FA
    done
fi
if [[ "$ONLY" == "invariant_o_proj_cca_unfold" ]]; then
    for batch in 1 64; do
        VLLM_TIDAR_BATCH_INVARIANT_O_PROJ=1 PATCH_PROBE_LAYER_HASH=1 \
            PATCH_PROBE_TRACE_MAX_EVENTS=1 \
            run_one invariant_o_proj_cca_unfold "$batch" 1 1 ROCM_AITER_FA
    done
fi
if [[ "$ONLY" == "invariant_dense" ]]; then
    for batch in 1 64; do
        VLLM_TIDAR_BATCH_INVARIANT_DENSE=1 PATCH_PROBE_LAYER_HASH=1 \
            PATCH_PROBE_TRACE_MAX_EVENTS=1 \
            run_one invariant_dense "$batch" 1 0 ROCM_AITER_FA
    done
fi
if [[ "$ONLY" == "invariant_dense_cca_unfold" ]]; then
    for batch in 1 64; do
        VLLM_TIDAR_BATCH_INVARIANT_DENSE=1 PATCH_PROBE_LAYER_HASH=1 \
            PATCH_PROBE_TRACE_MAX_EVENTS=1 \
            run_one invariant_dense_cca_unfold "$batch" 1 1 ROCM_AITER_FA
    done
fi

grep -H -E 'PATCH_PROBE_(CONTEXT|RESULT|TRACE)' "$LOGROOT"/*.log \
    >"$LOGROOT/results.txt" || true
echo "Logs: $LOGROOT"
