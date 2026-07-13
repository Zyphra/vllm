#!/usr/bin/env bash

set -euo pipefail

CKPT=${CKPT:-/data/checkpoints/smoediffusion_128k_64node-hf/iter_0012600}
DATA=${DATA:-benchmarks/tidar/aime25_zpo_texts.json}
LOGROOT=${LOGROOT:-/data/home/jinzhao/nv_v2_tidar_logs/lockstep_steady}
CONTAINER=${CONTAINER:-}
WORKDIR=${WORKDIR:-/data/home/jinzhao/workspace/vllm-smoe-amd-v024-mt10k}
PYTHON=${PYTHON:-python}
MAX_TOKENS=${MAX_TOKENS:-10000}
WARMUP_TOKENS=${WARMUP_TOKENS:-64}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-12000}
CASES=${CASES:?Set CASES to mode:batch:gpu entries}
DOCKER=${DOCKER:-docker}

mkdir -p "$LOGROOT"

common_env=(
    -e PYTHONPATH="$WORKDIR"
    -e VLLM_USE_V2_MODEL_RUNNER=1
    -e VLLM_ENABLE_V1_MULTIPROCESSING=0
    -e VLLM_ATTENTION_BACKEND=FLASH_ATTN
    -e VLLM_FLASH_ATTN_VERSION=3
    -e VLLM_SKIP_SDPA_PREINIT=1
    -e VLLM_CCA_TRITON=1
    -e VLLM_CCA_TRITON_FUSION_ENABLED=0
    -e VLLM_CCA_BATCH_INVARIANT_CONV=0
    -e VLLM_TIDAR_ROUTER_PAD=1
    -e PATCH_PROBE_STEADY=1
    -e PATCH_PROBE_EXACT_CUDAGRAPH_BATCH=1
)

run_case() {
    local mode=$1
    local batch=$2
    local gpu=$3
    local probe=benchmarks/tidar/probe_v2_ar.py
    local spec_args=()

    if [[ "$mode" == "tf" ]]; then
        probe=benchmarks/tidar/probe_v2_tidar_nv.py
        spec_args=(
            --draft-temp 0
            --num-spec-tokens 16
        )
    elif [[ "$mode" != "ar" ]]; then
        echo "Unknown mode: $mode" >&2
        return 2
    fi

    local tf_env=()
    local tf_env_direct=()
    if [[ "$mode" == "tf" ]]; then
        tf_env+=(
            -e VLLM_TIDAR_TWO_FORWARD=1
        )
        tf_env_direct+=(VLLM_TIDAR_TWO_FORWARD=1)
    fi

    if [[ -z "$CONTAINER" ]]; then
        (
            cd "$WORKDIR"
            env \
                PYTHONPATH="$WORKDIR" \
                VLLM_USE_V2_MODEL_RUNNER=1 \
                VLLM_ENABLE_V1_MULTIPROCESSING=0 \
                VLLM_ATTENTION_BACKEND=FLASH_ATTN \
                VLLM_FLASH_ATTN_VERSION=3 \
                VLLM_SKIP_SDPA_PREINIT=1 \
                VLLM_CCA_TRITON=1 \
                VLLM_CCA_TRITON_FUSION_ENABLED=0 \
                VLLM_CCA_BATCH_INVARIANT_CONV=0 \
                VLLM_TIDAR_ROUTER_PAD=1 \
                PATCH_PROBE_STEADY=1 \
                PATCH_PROBE_EXACT_CUDAGRAPH_BATCH=1 \
                PATCH_PROBE_STEADY_BATCH="$batch" \
                CUDA_VISIBLE_DEVICES="$gpu" \
                "${tf_env_direct[@]}" \
                "$PYTHON" -u "$probe" \
                --ckpt "$CKPT" --dataset "$DATA" --batch "$batch" \
                --num-prompts 1 --max-num-seqs "$batch" --n-sample "$batch" \
                --max-tokens "$MAX_TOKENS" --warmup-tokens "$WARMUP_TOKENS" \
                --repeats 1 --seed 0 --target-temp 0.6 \
                --max-model-len "$MAX_MODEL_LEN" \
                --max-num-batched-tokens 8192 --gpu-memory-utilization 0.65 \
                --backend FLASH_ATTN --cudagraph-mode FULL_AND_PIECEWISE \
                --prompt-token-ids --force-bos --ignore-eos \
                "${spec_args[@]}"
        ) >"$LOGROOT/${mode}_b${batch}.log" 2>&1
    else
        $DOCKER exec "${common_env[@]}" "${tf_env[@]}" \
            -e CUDA_VISIBLE_DEVICES="$gpu" \
            -e PATCH_PROBE_STEADY_BATCH="$batch" \
            "$CONTAINER" bash -lc \
            "cd '$WORKDIR' && python -u '$probe' \
             --ckpt '$CKPT' --dataset '$DATA' --batch '$batch' \
             --num-prompts 1 --max-num-seqs '$batch' --n-sample '$batch' \
             --max-tokens '$MAX_TOKENS' --warmup-tokens '$WARMUP_TOKENS' \
             --repeats 1 --seed 0 --target-temp 0.6 \
             --max-model-len '$MAX_MODEL_LEN' --max-num-batched-tokens 8192 \
             --gpu-memory-utilization 0.65 --backend FLASH_ATTN \
             --cudagraph-mode FULL_AND_PIECEWISE \
             --prompt-token-ids --force-bos --ignore-eos ${spec_args[*]}" \
            >"$LOGROOT/${mode}_b${batch}.log" 2>&1
    fi
}

pids=()
for case in $CASES; do
    IFS=: read -r mode batch gpu <<<"$case"
    run_case "$mode" "$batch" "$gpu" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

grep -H -E 'PATCH_PROBE_(CONTEXT|BEST)' "$LOGROOT"/*.log \
    >"$LOGROOT/results.txt" || true
cat "$LOGROOT/results.txt"
exit "$status"
