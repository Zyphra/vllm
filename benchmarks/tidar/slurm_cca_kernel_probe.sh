#!/usr/bin/env bash
#SBATCH --job-name=tidar-cca-probe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=00:10:00
#SBATCH --output=/shared/home/jinzhao/tfscope/slurm-tidar-cca-probe-%j.log

set -euo pipefail

REPO=${REPO:-/shared/home/jinzhao/tfscope/vllm-smoe-amd-bf16-v024}
IMAGE=${IMAGE:-zyphra/rocm-primus:aiter_pa_swa}
CONTAINER=tidar-cca-probe-${SLURM_JOB_ID}
GPU=${GPU:-1}

cleanup() {
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run --rm --name "$CONTAINER" --ipc=host --network=host \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES="$GPU" -e CUDA_VISIBLE_DEVICES="$GPU" \
    -v /shared:/shared -v "$REPO":/work -w /work \
    "$IMAGE" bash -lc '
        set -euo pipefail
        export PYTHONPATH=/work
        python -u benchmarks/tidar/probe_rocm_gemm_batch_shape.py \
            --label default
        TORCH_BLAS_PREFER_HIPBLASLT=1 \
            python -u benchmarks/tidar/probe_rocm_gemm_batch_shape.py \
            --label hipblaslt
        TORCH_BLAS_PREFER_HIPBLASLT=0 \
            python -u benchmarks/tidar/probe_rocm_gemm_batch_shape.py \
            --label rocblas
        TORCH_BLAS_PREFER_HIPBLASLT=0 ROCBLAS_DEFAULT_ATOMICS_MODE=0 \
            python -u benchmarks/tidar/probe_rocm_gemm_batch_shape.py \
            --label rocblas_no_atomics
        pip install -q pyzmq
        python -u benchmarks/tidar/probe_cca_batch_invariant.py
    '
