# Reproduce: TiDAR TF + FlashAttention on NVIDIA H100 (≈1906 tok/s)

Step-by-step to reproduce the **two-forward (TF) TiDAR on the `FLASH_ATTN`
backend** number on a single NVIDIA H100, using **Zyphra/vllm-smoe-amd**
`jinzhao/tidar_v016` — the same repo used for AMD, verified to build + run
TF+FA on CUDA. This is the fast NVIDIA path that AMD/ROCm cannot run today
(see [the perf report](amd_tidar_perf.md) — ROCm's `flash_attn` is the FA2
API; vLLM's backend needs the FA3-fork `vllm_flash_attn`).

Verified on `dgxh100-002` (vp-dgx-2), 2026-06-11.

## Result you should get

```
TOTAL: 357605 tokens / 187.60s = 1906.2 tok/s across 120 seqs
```

- **≈1906 tok/s**, spec-decode **mean acceptance ≈ 6.7–8.0**.
- AR baseline on the same box/bench (set `MODE=ar`): **≈1011 tok/s** — TF
  beats AR ~1.8× because FlashAttention makes the 2-forward verify cheap
  while acceptance stays ~7.

Confirm it is really TF + FA (not single-forward / not Flex) via this init
line in the log:

```
Initialized TiDAR self-speculation with a two-forward FlashAttention draft pass.
```

## Hardware / environment

- **1× H100 80GB**, single GPU — `tensor_parallel_size=1,
  data_parallel_size=1`.
- ckpt: `/data/checkpoints/smoediffusion_128k_64node-hf/iter_0012600`
  (HF-converted smoediffusion 128k / 64-node, iter 12600).
- prompts: `aime25_zpo_texts.json` — a JSON list of 30 AIME25 questions,
  **chat-templated** (already wrapped in the model's chat template).

### Build vllm-smoe-amd on CUDA

```bash
git clone --branch jinzhao/tidar_v016 git@github.com:Zyphra/vllm-smoe-amd.git
cd vllm-smoe-amd
python3.10 -m venv .venv && source .venv/bin/activate
pip install -U pip "setuptools>=77,<81" setuptools-scm wheel ninja cmake packaging jinja2 grpcio-tools==1.78.0
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128   # fork build-system pins torch 2.9.1
# from-source build (~30-40 min, sm_90). VLLM_USE_PRECOMPILED=0 is REQUIRED:
# the fork's custom SMoE CUDA kernels are not in any upstream precompiled wheel.
CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST="9.0" VLLM_USE_PRECOMPILED=0 MAX_JOBS=64 \
  pip install --no-build-isolation -e .
# place bench_match_sy.py + aime25_zpo_texts.json + the ckpt, then launch (below)
```

`setup.py` auto-detects CUDA (`torch.version.cuda`), so the same fork that
targets MI300X builds CUDA kernels + `vllm_flash_attn` on an H100. No
ROCm-only ops block the SMoE model on CUDA.

## Bench harness — `bench_match_sy.py`

```python
import time, os, json
import torch
from vllm import LLM, SamplingParams

def main():
    CKPT  = os.environ.get("CKPT",  "/data/checkpoints/smoediffusion_128k_64node-hf/iter_0012600")
    TEXTS = os.environ.get("TEXTS", "/data/groups/rl/jinzhao/workspace/tidar/aime25_zpo_texts.json")
    kwargs = dict(
        model=CKPT, dtype="bfloat16", gpu_memory_utilization=0.85,
        max_model_len=10000, max_num_seqs=16,          # b=16
        enforce_eager=False, seed=0, swap_space=4.0,
        attention_backend=os.environ.get("ATTN_BACKEND", "FLEX_ATTENTION"),
        disable_log_stats=False,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},   # captured
    )
    if os.environ.get("MODE", "tidar") == "tidar":
        kwargs["speculative_config"] = {"method": "tidar",
                                        "num_speculative_tokens": 16,
                                        "tidar_diff_temperature": 0.0}
    llm = LLM(**kwargs)
    prompts = json.load(open(TEXTS))[:int(os.environ.get("NPROMPTS", "30"))]
    # warmup
    llm.generate(prompts[:3], SamplingParams(n=1, temperature=0.5, max_tokens=50, seed=0), use_tqdm=False)
    torch.cuda.synchronize()
    sp = SamplingParams(n=4, temperature=0.5, max_tokens=8192, seed=0)
    t0 = time.perf_counter()
    out = llm.generate(prompts, sp, use_tqdm=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    toks = sum(len(o2.token_ids) for o in out for o2 in o.outputs)
    nseq = sum(len(o.outputs) for o in out)
    print("\nTOTAL: %d tokens / %.2fs = %.1f tok/s across %d seqs" % (toks, dt, toks/dt, nseq))
    print("=== BENCH DONE ===")

if __name__ == "__main__":
    main()
```

The bench hardcodes the measured config: `n=4`, `temperature=0.5`,
`max_tokens=8192`, `max_num_seqs=16` (b=16), captured
(`FULL_DECODE_ONLY`). Everything else is env-driven.

## Launch (the TF + FA combination)

```bash
VLLM_TIDAR_TWO_FORWARD=1 \      # TF mode  (unset / 0 = single-forward)
ATTN_BACKEND=FLASH_ATTN \      # the NVIDIA-only fast backend
MODE=tidar \
CKPT=/data/checkpoints/smoediffusion_128k_64node-hf/iter_0012600 \
TEXTS=/path/to/aime25_zpo_texts.json \
NPROMPTS=30 CUDA_VISIBLE_DEVICES=0 \
python -u bench_match_sy.py 2>&1 | tee nv_tf_fa.log
```

The two switches that define this run:

| env | value | meaning |
|---|---|---|
| `VLLM_TIDAR_TWO_FORWARD` | `1` | two-forward TiDAR (the 1906 path). Omit/0 → single-forward. |
| `ATTN_BACKEND` | `FLASH_ATTN` | FA3-fork backend. `FLEX_ATTENTION` on the same box is far slower. |

## Caveats

- The vllm-smoe-amd CUDA build is **from source** (~30-40 min on an H100,
  needs `nvcc`); `VLLM_USE_PRECOMPILED=0` is mandatory because the fork's
  custom SMoE CUDA kernels aren't in any upstream precompiled wheel.
- Single-GPU number. Per-GPU, not a DP=8 aggregate.
