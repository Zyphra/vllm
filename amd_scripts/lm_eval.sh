export HF_TOKEN=
export HF_HOME=

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export VLLM_CCA_TRITON=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_MHA=1
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
export VLLM_CCA_FUSED_ENABLED=1
export VLLM_CCA_FUSED_KERNEL_PATH=${SCRIPT_DIR}/../csrc/rocm/cca_decode_fused.cu
echo "VLLM_CCA_FUSED_KERNEL_PATH=${VLLM_CCA_FUSED_KERNEL_PATH}"
export VLLM_SMOE_FP8_A16=0

# export NCCL_IB_DISABLE=1
# export NCCL_NET_GDR_LEVEL=0
# export NCCL_SOCKET_IFNAME=

python -c "
import multiprocessing
multiprocessing.set_start_method('spawn')
import sys
sys.argv = [
    'lm_eval',
    '--model', 'vllm',
    '--model_args', 'pretrained=Zyphra-staging/smoe-midtrain_phase2_decay-30036,dtype=bfloat16,gpu_memory_utilization=0.9,enable_expert_parallel=True,data_parallel_size=2',
    '--batch_size', 'auto',
    '--trust_remote_code',
    '--cache_requests', 'true',
    '--tasks', 'gsm8k',
    '--log_samples',
    '--output_path', 'eval_results',
]
from lm_eval.__main__ import cli_evaluate
cli_evaluate()
"
