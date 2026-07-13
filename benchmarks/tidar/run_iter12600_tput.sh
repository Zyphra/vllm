#!/usr/bin/env bash
set -euo pipefail

CKPT=${CKPT:?Set CKPT to the iter_0012600 checkpoint directory}
DATA=${DATA:-benchmarks/tidar/aime25_zpo_texts.json}
BACKEND=${BACKEND:-ROCM_AITER_FA}
GPU=${GPU:-1}
BATCHES=${BATCHES:-"1 8 16 64"}
MAX_TOKENS=${MAX_TOKENS:-2000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-12000}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
WARMUP_TOKENS=${WARMUP_TOKENS:-64}
TARGET_TEMP=${TARGET_TEMP:-0.0}
DRAFT_TEMP=${DRAFT_TEMP:-0.0}
SEED=${SEED:-0}
N_SAMPLE_AR=${N_SAMPLE_AR:-1}
N_SAMPLE_TF=${N_SAMPLE_TF:-10}
RUN_AR=${RUN_AR:-1}
RUN_TF=${RUN_TF:-1}
LOGROOT=${LOGROOT:-"tidar_iter12600_tput_$(date +%Y%m%d_%H%M%S)"}

mkdir -p "$LOGROOT"

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ATTENTION_BACKEND="$BACKEND"
export VLLM_SKIP_SDPA_PREINIT=1
export VLLM_CCA_TRITON=1
export VLLM_CCA_TRITON_FUSION_ENABLED=0
export VLLM_CCA_AMD_CONV_UNFOLD=0
export VLLM_TIDAR_ROUTER_PAD=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_MOE_PADDING=1
export VLLM_TIDAR_SMOE_MOE_OP=0
export VLLM_SMOE_ROCM_BF16_LM_HEAD=0

if [[ "$BACKEND" == "FLASH_ATTN" ]]; then
    export VLLM_FLASH_ATTN_VERSION=${VLLM_FLASH_ATTN_VERSION:-3}
fi

run_ar() {
    local batch=$1
    local log="$LOGROOT/ar_b${batch}.log"
    HIP_VISIBLE_DEVICES="$GPU" CUDA_VISIBLE_DEVICES="$GPU" \
    python -u benchmarks/tidar/probe_v2_ar.py \
        --ckpt "$CKPT" --dataset "$DATA" \
        --batch "$batch" --num-prompts "$batch" --max-num-seqs "$batch" \
        --n-sample "$N_SAMPLE_AR" --max-tokens "$MAX_TOKENS" \
        --warmup-tokens "$WARMUP_TOKENS" --repeats 1 \
        --seed "$SEED" \
        --target-temp "$TARGET_TEMP" --max-model-len "$MAX_MODEL_LEN" \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --backend "$BACKEND" --cudagraph-mode FULL_AND_PIECEWISE \
        --prompt-token-ids --force-bos --ignore-eos 2>&1 | tee "$log"
}

run_tf() {
    local batch=$1
    local log="$LOGROOT/tf_b${batch}.log"
    HIP_VISIBLE_DEVICES="$GPU" CUDA_VISIBLE_DEVICES="$GPU" \
    VLLM_TIDAR_TWO_FORWARD=1 \
    VLLM_TIDAR_USE_TF_PAGED_ATTENTION=1 \
    VLLM_TIDAR_TF_PAGED_NO_SPLITS=1 \
    VLLM_TIDAR_FA_NO_SPLITS=1 \
    python -u benchmarks/tidar/probe_v2_tidar_nv.py \
        --ckpt "$CKPT" --dataset "$DATA" \
        --batch "$batch" --num-prompts "$batch" --max-num-seqs "$batch" \
        --n-sample "$N_SAMPLE_TF" --max-tokens "$MAX_TOKENS" \
        --warmup-tokens "$WARMUP_TOKENS" --repeats 1 \
        --seed "$SEED" \
        --target-temp "$TARGET_TEMP" --draft-temp "$DRAFT_TEMP" \
        --num-spec-tokens 16 \
        --max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens 8192 \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --backend "$BACKEND" --cudagraph-mode FULL_AND_PIECEWISE \
        --prompt-token-ids --force-bos --ignore-eos \
        2>&1 | tee "$log"
}

for batch in $BATCHES; do
    echo "PAIR_START batch=$batch $(date -Iseconds)" | tee -a "$LOGROOT/summary.log"
    if [[ "$RUN_AR" == "1" ]]; then
        run_ar "$batch"
    fi
    if [[ "$RUN_TF" == "1" ]]; then
        run_tf "$batch"
    fi
    echo "PAIR_END batch=$batch $(date -Iseconds)" | tee -a "$LOGROOT/summary.log"
done

grep -H -E 'PATCH_PROBE_(CONTEXT|BEST)' "$LOGROOT"/*.log \
    > "$LOGROOT/results.txt"
cat "$LOGROOT/results.txt"
