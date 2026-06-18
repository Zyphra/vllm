#!/usr/bin/env python
"""Turnkey reproducer for the AMD MI300X TiDAR single-forward (SF) throughput
numbers in docs/amd_tidar_perf.md.

    python benchmarks/tidar/bench_amd_sf.py --ckpt /path/to/iter_0012600
    python benchmarks/tidar/bench_amd_sf.py --ckpt /path/... --dense

Expected (MI300X, b=16, captured):
    [0,4,7,11]  -> ~760-800 tok/s, true-mean accept ~5.5
    --dense     -> ~510-545 tok/s, true-mean accept ~7.3

This script *forces* every knob that selects the fast code path (ENV block
below) BEFORE importing vllm, so you cannot land on the slow path by
forgetting an env var. The only thing you must supply is --ckpt.

If you previously got ~10% of the expected throughput, it is almost always
ONE of these — all of which this script now pins:
  * attention backend != FLEX_ATTENTION      -> SF silently degrades to AR speed
  * proposal levels left at default (4,7,10)  -> acceptance collapses to ~1.0
  * ran eager instead of captured             -> no cudagraph
Confirm from the engine log that you see, near startup:
    Using FlexAttention backend
    TiDAR single-forward mode ENABLED with K=16, P=..., acc_levels=(0, 4, 7, 11)
and that the SpecDecoding per-position accept is ~0.8 decaying to ~0.2 (NOT a
flat ~1.0). If throughput is < 150 tok/s the script prints a loud warning.
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=os.environ.get("CKPT"),
                    help="path to a smoediffusion HF checkpoint (e.g. iter_0012600). REQUIRED.")
parser.add_argument("--dense", action="store_true",
                    help="dense proposal levels 0..16 (default: [0,4,7,11])")
parser.add_argument("--levels", default=None,
                    help="comma-separated proposal acc levels (overrides --dense)")
parser.add_argument("--prompts", default=os.environ.get(
    "PROMPTS", os.path.join(os.path.dirname(__file__), "aime25_zpo_texts.json")))
parser.add_argument("--nprompts", type=int, default=int(os.environ.get("NPROMPTS", "30")))
parser.add_argument("--n", type=int, default=int(os.environ.get("N", "4")))
parser.add_argument("--temp", type=float, default=float(os.environ.get("TEMP", "0.5")))
parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAXTOK", "8192")))
parser.add_argument("--batch", type=int, default=int(os.environ.get("MAXSEQS", "16")))
parser.add_argument("--mml", type=int, default=int(os.environ.get("MML", "10000")))
parser.add_argument("--eager", action="store_true", help="disable cudagraph (slow; debug only)")
args = parser.parse_args()

if not args.ckpt:
    sys.exit("ERROR: pass --ckpt /path/to/smoediffusion/checkpoint "
             "(the doc used iter_0012600). No default — supply your own copy.")

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
print("TiDAR SF reproducer — forced config")
print(f"  VLLM_ATTENTION_BACKEND         = {os.environ['VLLM_ATTENTION_BACKEND']}")
print(f"  VLLM_TIDAR_SF_TRITON           = {os.environ['VLLM_TIDAR_SF_TRITON']}")
print(f"  VLLM_TIDAR_PROPOSAL_ACC_LEVELS = {LEVELS}")
print(f"  captured                       = {not args.eager} (FULL_DECODE_ONLY)")
print(f"  ckpt                           = {args.ckpt}")
print(f"  batch={args.batch} n={args.n} temp={args.temp} max_tokens={args.max_tokens}")
print("=" * 72)

import json
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
        attention_backend="FLEX_ATTENTION",  # belt + suspenders (env is ignored on ROCm)
        disable_log_stats=False,
        speculative_config={
            "method": "tidar",
            "num_speculative_tokens": 16,
            "tidar_diff_temperature": 0.0,
        },
        **({} if args.eager else {"compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}}),
    )
    llm = LLM(**kwargs)

    with open(args.prompts) as f:
        prompts = json.load(f)[: args.nprompts]
    print(f"{len(prompts)} prompts loaded")

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
    print(f"\nTOTAL: {total} tokens / {dt:.2f}s = {tps:.1f} tok/s across {nseqs} seqs")
    print(f"len p50={lens[len(lens) // 2]} p90={lens[int(len(lens) * 0.9)]} max={lens[-1]}")
    print("=== BENCH DONE ===")
    if tps < 150:
        print(
            "\n*** WARNING: throughput < 150 tok/s at b=16 — you are almost certainly NOT on\n"
            "    the SF Triton path. Check the engine log for these two lines:\n"
            "      'Using FlexAttention backend'   (NOT AITER/Triton/Unified)\n"
            f"      'single-forward mode ENABLED ... acc_levels=({LEVELS.replace(',', ', ')})'\n"
            "    and per-position accept ~0.8->0.2 (a flat ~1.0 means levels collapsed).",
            file=sys.stderr)


if __name__ == "__main__":
    main()
