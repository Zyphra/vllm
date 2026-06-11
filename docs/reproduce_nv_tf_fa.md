# Reproduce: TiDAR TF + FlashAttention on NVIDIA H100 (≈1827 tok/s)

Step-by-step to reproduce the headline **two-forward (TF) TiDAR on the
`FLASH_ATTN` backend** number on a single NVIDIA H100. This is the fast
NVIDIA path that AMD/ROCm cannot run today (see [the perf
report](amd_tidar_perf.md) — ROCm's `flash_attn` is the FA2 API; vLLM's
backend needs the FA3-fork `vllm_flash_attn`).

Verified on `dgxh100-002` (vp-dgx-2), run on 2026-06-11.

## Result you should get

```
TOTAL: 386915 tokens / 211.82s = 1826.6 tok/s across 120 seqs
```

- **≈1827 tok/s**, spec-decode **mean acceptance ≈ 6.99**.
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
- vLLM: **Zyphra/Zvllm**, branch **`jinzhao/tidar`** (TiDAR
  self-speculation), built **with `vllm_flash_attn`** (the FA3 fork — this
  is what provides the `FLASH_ATTN` backend). Logged run was vLLM
  `v0.16.1.dev37+gb106800a0`.
- ckpt: `/data/checkpoints/smoediffusion_128k_64node-hf/iter_0012600`
  (HF-converted smoediffusion 128k / 64-node, iter 12600).
- prompts: `aime25_zpo_texts.json` — a JSON list of 30 AIME25 questions,
  **chat-templated** (already wrapped in the model's chat template).

### On vp-dgx-2 the env is already built

```
/data/home/jinzhao/workspace/tidar/.venv          # venv (python3.10)
/data/home/jinzhao/workspace/tidar/Zvllm-sf-fixed  # vLLM editable source
/data/home/jinzhao/workspace/tidar/bench_match_sy.py
/data/home/jinzhao/workspace/tidar/aime25_zpo_texts.json
```

### From scratch on a fresh H100 box

```bash
git clone git@github.com:Zyphra/Zvllm.git && cd Zvllm
git checkout jinzhao/tidar
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -e .            # builds vllm_flash_attn (FA3 fork) for the FLASH_ATTN backend
# place bench_match_sy.py + aime25_zpo_texts.json + the ckpt, then launch (below)
```

## Bench harness — `bench_match_sy.py`

```python
import time, os, json
import torch
from vllm import LLM, SamplingParams

def main():
    CKPT  = os.environ.get("CKPT",  "/data/checkpoints/smoediffusion_128k_64node-hf/iter_0012600")
    TEXTS = os.environ.get("TEXTS", "/data/home/jinzhao/workspace/tidar/aime25_zpo_texts.json")
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
cd /data/home/jinzhao/workspace/tidar
VLLM_TIDAR_TWO_FORWARD=1 \      # TF mode  (unset / 0 = single-forward)
ATTN_BACKEND=FLASH_ATTN \      # the NVIDIA-only fast backend
MODE=tidar \
CKPT=/data/checkpoints/smoediffusion_128k_64node-hf/iter_0012600 \
TEXTS=/data/home/jinzhao/workspace/tidar/aime25_zpo_texts.json \
NPROMPTS=30 CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -u bench_match_sy.py 2>&1 | tee nv_tf_fa.log
```

The two switches that define this run:

| env | value | meaning |
|---|---|---|
| `VLLM_TIDAR_TWO_FORWARD` | `1` | two-forward TiDAR (the 1827 path). Omit/0 → single-forward. |
| `ATTN_BACKEND` | `FLASH_ATTN` | FA3-fork backend. `FLEX_ATTENTION` on the same box is far slower. |

## Caveats

- The vp-dgx-2 `Zvllm-sf-fixed` checkout is currently on branch
  `jinzhao/tidar_sf_tfna0`, not the `jinzhao/tidar` @ `gb106800a0` that
  produced the logged 1827. TF+FA is stable across these TiDAR branches,
  but for a bit-exact match `git -C Zvllm-sf-fixed checkout jinzhao/tidar`
  first (re-run `uv pip install -e .` if the build complains).
- This is single-GPU. The colleague's quoted NVIDIA throughput is per-GPU,
  not a DP=8 aggregate.
