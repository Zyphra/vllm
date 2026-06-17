#!/usr/bin/env python
"""Reproduce the AMD MI300X TiDAR single-forward (SF) throughput numbers from
docs/amd_tidar_perf.md.

Two headline configs (selected via VLLM_TIDAR_PROPOSAL_ACC_LEVELS):
  * SF [0,4,7,11]  (P=4)   -> ~803 tok/s, accept ~5.6   (fastest)
  * SF [0..16]     (P=17)  -> ~544 tok/s, accept ~7.6   (highest accept)

The only knob that changes between the two is the proposal-level set; every
other parameter below is the shared "matched config" the doc benchmarks use.

Run via the runbook in docs/amd_tidar_perf.md (must be FLEX_ATTENTION +
VLLM_TIDAR_SF_TRITON=1 + captured). Environment overrides (all optional):
  CKPT       model path           (default: doc checkpoint iter_0012600)
  PROMPTS    chat-template prompts json (list[str])
  NPROMPTS   number of prompts    (default 30)
  N          samples per prompt   (default 4)
  TEMP       sampling temperature (default 0.5)
  MAXTOK     max new tokens       (default 8192)
  MAXSEQS    max_num_seqs / batch (default 16)
  MML        max_model_len        (default 10000)
  EAGER=1    disable cudagraph capture (default: captured FULL_DECODE_ONLY)
  ATTN_BACKEND  attention backend (default FLEX_ATTENTION; SF needs Flex)
  MODE=ar    run plain autoregressive instead of tidar SF
"""
import json
import os
import re
import time

import torch

from vllm import LLM, SamplingParams

CKPT = os.environ.get(
    "CKPT",
    "/shared/home/henry/checkpoints/hf/smoediffusion_128k_64node/iter_0012600",
)
PROMPTS = os.environ.get(
    "PROMPTS", os.path.join(os.path.dirname(__file__), "aime25_zpo_texts.json")
)


def true_mean_accept(log_text: str, k: int = 16) -> float:
    """1 + sum(accepted)/sum(drafted/K) over ALL SpecDecoding windows."""
    acc = re.findall(r"Accepted: (\d+) tokens", log_text)
    drf = re.findall(r"Drafted: (\d+) tokens", log_text)
    if not acc or not drf:
        return float("nan")
    a = sum(int(x) for x in acc)
    d = sum(int(x) for x in drf)
    return 1 + a / (d / k) if d else float("nan")


def main():
    eager = os.environ.get("EAGER", "0") == "1"
    kwargs = dict(
        model=CKPT,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=int(os.environ.get("MML", "10000")),
        max_num_seqs=int(os.environ.get("MAXSEQS", "16")),
        enforce_eager=eager,
        seed=0,
        swap_space=4.0,
        attention_backend=os.environ.get("ATTN_BACKEND", "FLEX_ATTENTION"),
        disable_log_stats=False,
        **({} if eager else {"compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}}),
    )
    if os.environ.get("MODE", "tidar") == "tidar":
        kwargs["speculative_config"] = {
            "method": "tidar",
            "num_speculative_tokens": 16,
            "tidar_diff_temperature": 0.0,
        }
    print(f"levels={os.environ.get('VLLM_TIDAR_PROPOSAL_ACC_LEVELS', '(default)')} "
          f"backend={kwargs['attention_backend']} eager={eager}")
    llm = LLM(**kwargs)

    with open(PROMPTS) as f:
        prompts = json.load(f)
    prompts = prompts[: int(os.environ.get("NPROMPTS", "30"))]
    print(f"{len(prompts)} prompts loaded")

    # short warmup so capture / first-forward compile is excluded from timing
    llm.generate(prompts[:3], SamplingParams(n=1, temperature=0.5, max_tokens=50, seed=0),
                 use_tqdm=False)
    torch.cuda.synchronize()

    sp = SamplingParams(
        n=int(os.environ.get("N", "4")),
        temperature=float(os.environ.get("TEMP", "0.5")),
        max_tokens=int(os.environ.get("MAXTOK", "8192")),
        seed=0,
    )
    t0 = time.perf_counter()
    out = llm.generate(prompts, sp, use_tqdm=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    total = sum(len(o2.token_ids) for o in out for o2 in o.outputs)
    nseqs = sum(len(o.outputs) for o in out)
    lens = sorted(len(o2.token_ids) for o in out for o2 in o.outputs)
    print(f"\nTOTAL: {total} tokens / {dt:.2f}s = {total / dt:.1f} tok/s across {nseqs} seqs")
    print(f"len p50={lens[len(lens) // 2]} p90={lens[int(len(lens) * 0.9)]} max={lens[-1]}")
    print("=== BENCH DONE ===")


if __name__ == "__main__":
    main()
