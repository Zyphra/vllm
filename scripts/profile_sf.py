"""Profile SF FULL captured to break down per-component time.

Captures one prompt's worth of SF steps with torch.profiler and dumps
the top CUDA ops by time. Tells us where the 22ms/step is going.
"""
import os
import time

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset


CKPT = "/data/checkpoints/smoediffusion_128k_64node-hf/iter_0012000"


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    ds = load_dataset("MathArena/aime_2025", split="train[:1]")
    problem = ds[0]["problem"]

    llm = LLM(
        model=CKPT, dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096, max_num_seqs=1,
        enforce_eager=False, seed=0,
        disable_log_stats=True, swap_space=16.0,
        attention_backend="FLEX_ATTENTION",
        speculative_config={
            "method": "tidar",
            "num_speculative_tokens": 16,
            "tidar_diff_temperature": 0.0,
        },
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
    )

    chat = [{"role": "user", "content": problem}]
    text = tok.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)

    sp = SamplingParams(n=1, temperature=0.0, max_tokens=200, seed=0)
    torch.cuda.synchronize()

    # profile a short run
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        t0 = time.perf_counter()
        out = llm.generate([text], sp, use_tqdm=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    toks = out[0].outputs[0].token_ids
    print(f"\n=== PROFILE === {len(toks)}t in {dt:.2f}s = "
          f"{len(toks)/dt:.1f} tok/s")

    # top CUDA ops by total time
    print("\nTop 30 ops by self_cuda_time_total:")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=30,
        max_name_column_width=80,
    ))
    print("\nTop 30 ops by cuda_time_total:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=30,
        max_name_column_width=80,
    ))


if __name__ == "__main__":
    main()
