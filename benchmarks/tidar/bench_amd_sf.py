#!/usr/bin/env python
"""Self-contained reproducer for the AMD MI300X TiDAR single-forward (SF)
throughput numbers in docs/amd_tidar_perf.md.

ONE file, no companion data: the 30 AIME25 chat-template prompts are embedded
below. Drop it anywhere inside a built vllm-smoe-amd checkout and run:

    python bench_amd_sf.py --ckpt /path/to/iter_0012600           # ~760-800 tok/s
    python bench_amd_sf.py --ckpt /path/to/iter_0012600 --dense   # ~510-545 tok/s

It FORCES every knob that selects the fast code path (ENV block below) BEFORE
importing vllm, so you cannot land on the slow path by forgetting an env var.
The only thing you supply is --ckpt (your own smoediffusion checkpoint).

If you previously got ~10% of the expected throughput you were not on the SF
Triton path: either backend != FLEX_ATTENTION (SF degrades to AR speed),
proposal levels left at the default (4,7,10) (accept collapses to ~1.0), or it
ran eager not captured. This script pins all three; it warns if tok/s < 150
and names the log line to check.
"""
import argparse
import base64
import gzip
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=os.environ.get("CKPT"),
                    help="path to a smoediffusion HF checkpoint (e.g. iter_0012600). REQUIRED.")
parser.add_argument("--dense", action="store_true",
                    help="dense proposal levels 0..16 (default: [0,4,7,11])")
parser.add_argument("--levels", default=None,
                    help="comma-separated proposal acc levels (overrides --dense)")
parser.add_argument("--nprompts", type=int, default=30)
parser.add_argument("--n", type=int, default=4)
parser.add_argument("--temp", type=float, default=0.5)
parser.add_argument("--max-tokens", type=int, default=8192)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--mml", type=int, default=10000)
parser.add_argument("--eager", action="store_true", help="disable cudagraph (slow; debug only)")
args = parser.parse_args()

if not args.ckpt:
    sys.exit("ERROR: pass --ckpt /path/to/smoediffusion/checkpoint "
             "(the doc used iter_0012600). No default - supply your own copy.")

if args.levels:
    LEVELS = args.levels
elif args.dense:
    LEVELS = ",".join(str(i) for i in range(17))   # 0..16
else:
    LEVELS = "0,4,7,11"

# ---- forced environment (MUST be set before `import vllm`) -----------------
os.environ["VLLM_ATTENTION_BACKEND"] = "FLEX_ATTENTION"   # SF needs Flex
os.environ["VLLM_TIDAR_SF_TRITON"] = "1"                  # paged Triton kernel (the AMD win)
os.environ["VLLM_TIDAR_PROPOSAL_ACC_LEVELS"] = LEVELS     # MUST include 0
os.environ["VLLM_SKIP_SDPA_PREINIT"] = "1"                # avoid intermittent import segfault
os.environ["VLLM_CCA_TRITON"] = "1"                       # Triton CCA (capture-safe on ROCm)

print("=" * 72)
print("TiDAR SF reproducer (self-contained) - forced config")
print("  VLLM_ATTENTION_BACKEND         =", os.environ["VLLM_ATTENTION_BACKEND"])
print("  VLLM_TIDAR_SF_TRITON           =", os.environ["VLLM_TIDAR_SF_TRITON"])
print("  VLLM_TIDAR_PROPOSAL_ACC_LEVELS =", LEVELS)
print("  captured                       =", (not args.eager), "(FULL_DECODE_ONLY)")
print("  ckpt                           =", args.ckpt)
print("  batch=%d n=%d temp=%s max_tokens=%d" % (args.batch, args.n, args.temp, args.max_tokens))
print("=" * 72)

# 30 AIME25 problems, chat-templated (gzip+base64 of aime25_zpo_texts.json).
# Inspect: python -c "import bench_amd_sf as b; print(b.PROMPTS[0])"
_PROMPTS_B64 = (
    "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"
)
PROMPTS = json.loads(gzip.decompress(base64.b64decode("".join(_PROMPTS_B64))).decode())

import time
import torch
from vllm import LLM, SamplingParams


def main():
    kwargs = dict(
        model=args.ckpt,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=args.mml,
        max_num_seqs=args.batch,
        enforce_eager=args.eager,
        seed=0,
        swap_space=4.0,
        attention_backend="FLEX_ATTENTION",  # belt + suspenders (env ignored on ROCm)
        disable_log_stats=False,
        speculative_config={
            "method": "tidar",
            "num_speculative_tokens": 16,
            "tidar_diff_temperature": 0.0,
        },
        **({} if args.eager else {"compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}}),
    )
    llm = LLM(**kwargs)
    prompts = PROMPTS[: args.nprompts]
    print(len(prompts), "prompts loaded")

    # warmup excluded from timing (captures graphs / compiles kernels)
    llm.generate(prompts[:3], SamplingParams(n=1, temperature=0.5, max_tokens=50, seed=0),
                 use_tqdm=False)
    torch.cuda.synchronize()

    sp = SamplingParams(n=args.n, temperature=args.temp, max_tokens=args.max_tokens, seed=0)
    t0 = time.perf_counter()
    out = llm.generate(prompts, sp, use_tqdm=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    total = sum(len(o2.token_ids) for o in out for o2 in o.outputs)
    nseqs = sum(len(o.outputs) for o in out)
    lens = sorted(len(o2.token_ids) for o in out for o2 in o.outputs)
    tps = total / dt
    print("\nTOTAL: %d tokens / %.2fs = %.1f tok/s across %d seqs" % (total, dt, tps, nseqs))
    print("len p50=%d p90=%d max=%d" % (lens[len(lens) // 2], lens[int(len(lens) * 0.9)], lens[-1]))
    print("=== BENCH DONE ===")
    if tps < 150:
        sys.stderr.write(
            "\n*** WARNING: throughput < 150 tok/s at b=16 - you are almost certainly NOT\n"
            "    on the SF Triton path. Check the engine log for:\n"
            "      'Using FlexAttention backend'   (NOT AITER/Triton/Unified)\n"
            "      'single-forward mode ENABLED ... acc_levels=(" + LEVELS.replace(",", ", ") + ")'\n"
            "    and per-position accept ~0.8->0.2 (a flat ~1.0 means levels collapsed).\n")


if __name__ == "__main__":
    main()
