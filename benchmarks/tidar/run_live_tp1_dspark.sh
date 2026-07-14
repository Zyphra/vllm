#!/usr/bin/env bash

set -euo pipefail

ACTION=${1:-serve}
MODEL=${MODEL:-/data/checkpoints/SMOE_DIFFUSION_sftv2_tv_dspark+markov-hf/iter_0005600}
WORKLOAD=${WORKLOAD:-/data/home/jinzhao/tidar_live_tp1_20260713/workload_16x8.jsonl}
LOG_DIR=${LOG_DIR:-/data/home/jinzhao/tidar_live_tp1_20260713}
PORT=${PORT:-8107}
SERVED_MODEL=${SERVED_MODEL:-dspark-markov-iter5600}
PYTHON=${PYTHON:-python3}

mkdir -p "$LOG_DIR"

case "$ACTION" in
    serve)
        export PYTHONPATH=${PYTHONPATH:-/work}
        export VLLM_USE_V1=1
        export VLLM_USE_V2_MODEL_RUNNER=1
        export VLLM_ENABLE_V1_MULTIPROCESSING=0
        export VLLM_ATTENTION_BACKEND=FLASH_ATTN
        export VLLM_FLASH_ATTN_VERSION=3
        export VLLM_ROCM_MOE_PADDING=1
        export VLLM_CACHE_ROOT=/tmp/vllm_cache
        export VLLM_SOURCE_ROOT=${VLLM_SOURCE_ROOT:-/work}
        export VLLM_SRC_ROOT=${VLLM_SRC_ROOT:-/work}
        export VLLM_CCA_KERNEL_ROOT=${VLLM_CCA_KERNEL_ROOT:-/work}
        export VLLM_CCA_TRITON=1
        export VLLM_CCA_FUSED_ENABLED=0
        export VLLM_CCA_TRITON_FUSION_ENABLED=0
        export VLLM_TIDAR_TWO_FORWARD=1
        export VLLM_TIDAR_ROUTER_PAD=1
        export VLLM_TIDAR_FA_NO_SPLITS=1
        export VLLM_TIDAR_DP_EAGER_DRAFT=1
        export VLLM_TIDAR_RETURN_AR_LOGPROBS=1
        export VLLM_TIDAR_AR_TEMPERATURE=0.6
        export VLLM_TIDAR_DSPARK=1
        export VLLM_DSPARK_GLOBAL_RESET=1
        export VLLM_DSPARK_NO_MARKOV=0
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        exec "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
            --host 0.0.0.0 \
            --port "$PORT" \
            --served-model-name "$SERVED_MODEL" \
            --dtype bfloat16 \
            --load-format auto \
            --max-model-len 33824 \
            --max-num-seqs 32 \
            --enable-chunked-prefill \
            --max-num-batched-tokens 262144 \
            --disable-custom-all-reduce \
            --gpu-memory-utilization 0.55 \
            --tensor-parallel-size 1 \
            --seed 0 \
            --scheduling-policy priority \
            --compilation-config \
                '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":"VLLM_COMPILE","custom_ops":["all","-rms_norm"]}' \
            --mamba-cache-dtype auto \
            --mamba-cache-mode none \
            --enable-return-routed-experts \
            --speculative-config \
                '{"method":"tidar","num_speculative_tokens":16,"tidar_diff_temperature":0.0,"tidar_ar_temperature":0.6}' \
            --override-generation-config \
                '{"temperature":0.6,"top_k":-1,"top_p":1.0,"repetition_penalty":1.0,"max_new_tokens":32768}'
        ;;
    bench)
        export PYTHONPATH=${PYTHONPATH:-/work}
        exec "$PYTHON" -m vllm.entrypoints.cli.main bench serve \
            --backend vllm \
            --base-url "http://127.0.0.1:$PORT" \
            --endpoint /v1/completions \
            --model "$SERVED_MODEL" \
            --tokenizer "$MODEL" \
            --dataset-name custom \
            --dataset-path "$WORKLOAD" \
            --skip-chat-template \
            --disable-shuffle \
            --no-oversample \
            --num-prompts 128 \
            --custom-output-len 32768 \
            --max-concurrency 32 \
            --request-rate inf \
            --temperature 0.6 \
            --top-p 1.0 \
            --top-k -1 \
            --repetition-penalty 1.0 \
            --logprobs 1 \
            --extra-body '{"add_special_tokens":false}' \
            --percentile-metrics ttft,tpot,itl,e2el \
            --save-result \
            --save-detailed \
            --result-dir "$LOG_DIR" \
            --result-filename result.json
        ;;
    *)
        echo "Usage: $0 {serve|bench}" >&2
        exit 2
        ;;
esac
