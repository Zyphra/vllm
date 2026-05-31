"""SF / TF throughput + acceptance sweep on 16 prompts.

Runs one config at a time (set via env / CLI). Prints aggregate avg tok/s
and -- when SF spec decoding is enabled -- vLLM's reported acceptance.

Datasets:
    --dataset mmlu_pro   -> TIGER-Lab/MMLU-Pro (test), multiple choice
    --dataset aime25     -> AIME 2025 (parquet, 30 problems)
    --dataset aime26     -> HuggingFaceH4/aime_2024 (test) or Maxwell-Jia/AIME_2024
                            (problem -> reasoning -> integer)
    --dataset hmmt       -> /data/datasets/zpo/hmmt.parquet (HMMT 2026 set)

Thinking:
    --thinking on  -> use the model's chat template with thinking on
    --thinking off -> use the chat template with thinking off

Batch:
    --batch N      -> set max_num_seqs and feed all prompts to a single
                       llm.generate() call (vLLM schedules them in batches).

Captures (cudagraph):
    --explicit-captures -> compute an SF-aware capture set
                           (Kp1, b*Kp1 for b in 1..batch, b*sf_per_req
                           for b in 1..batch, filtered to avoid mamba_attn
                           uniform-batch assertion). Default uses vLLM auto-detect.

Mode selection:
    SF (single-forward) is the default since the tidar_TF + tidar_SF merge.
    Opt into TF with `VLLM_TIDAR_TWO_FORWARD=1`.

Run a single config (SF — default mode):

    VLLM_TIDAR_PROPOSAL_ACC_LEVELS=0,5 \\
    VLLM_ATTENTION_BACKEND=FLEX_ATTENTION \\
    VLLM_TIDAR_SF_TRITON=1 \\
    python scripts/_sf_mmlu_sweep.py \\
        --ckpt /path/to/ckpt --dataset aime26 --thinking off \\
        --batch 8 --K 16 --n 16 --max-tokens 1024 --explicit-captures \\
        --tag sf-p2-0-5

Or opt into TF:

    VLLM_TIDAR_TWO_FORWARD=1 python scripts/_sf_mmlu_sweep.py ...
"""

import argparse
import json
import os
import time
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def mmlu_pro_prompt(question: str, options: list) -> str:
    letters = "ABCDEFGHIJ"
    opt_block = "\n".join(f"{letters[i]}. {opt}"
                          for i, opt in enumerate(options))
    return (
        f"The following is a multiple choice question. Reason step by step "
        f"and then write 'The answer is (X)' where X is the letter of the "
        f"correct option.\n\n"
        f"Question: {question}\n"
        f"Options:\n{opt_block}\n\n"
        f"Let's think step by step.\n")


def aime26_prompt(problem: str) -> str:
    return (
        f"{problem}\n\nReason step by step and put your final answer as an "
        f"integer in \\boxed{{}}.\n")


def build_llm(ckpt: str, mode: str, K: int, t_diff: float,
              max_model_len: int, eager: bool, gpu_mem: float,
              batch: int = 1, explicit_captures: bool = False):
    Kp1 = K + 1
    kwargs = dict(
        model=ckpt, dtype="bfloat16",
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len, max_num_seqs=batch,
        enforce_eager=eager,
        seed=0, disable_log_stats=False,
        swap_space=16.0,
    )
    # v0.16 removed env var auto-selection; honor VLLM_ATTENTION_BACKEND
    # here for backward compatibility with v0.15.x recipes.
    _backend_env = os.environ.get("VLLM_ATTENTION_BACKEND")
    if _backend_env:
        kwargs["attention_backend"] = _backend_env
    if mode == "tidar":
        kwargs["speculative_config"] = {
            "method": "tidar",
            "num_speculative_tokens": K,
            "tidar_diff_temperature": t_diff,
        }
        if not eager:
            if explicit_captures:
                # SF capture sizes: verify segment + full SF per req.
                # Layout depends on VLLM_TIDAR_NO_BONUS:
                #   default  -> verify=K+1, proposal_seg_len=K+1,
                #               sf_per_req = K+1 + P*(K+1) = (P+1)*(K+1).
                #   no_bonus -> verify=K,   proposal_seg_len=K,
                #               sf_per_req = K   + P*K     = (P+1)*K.
                # Verify spec_sizes are kept at b*Kp1 for the TF capture
                # path; SF spec capture pinpoints sf_per_req multiples.
                lvls = os.environ.get(
                    "VLLM_TIDAR_PROPOSAL_ACC_LEVELS", "")
                P = (len([x for x in lvls.split(",") if x.strip()])
                     if lvls.strip() else 3)
                # ZAP-ONLY: K+1 layout always.
                sf_per_req = Kp1 + P * Kp1
                spec_sizes = [b * Kp1 for b in range(1, batch + 1)]
                spec_filtered = [s for s in spec_sizes
                                 if s < sf_per_req or s % sf_per_req == 0]
                sf_sizes = [b * sf_per_req for b in range(1, batch + 1)]
                explicit = sorted(set(spec_filtered) | set(sf_sizes))
                kwargs["compilation_config"] = {
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": explicit,
                }
            else:
                kwargs["compilation_config"] = {
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                }
    else:  # ar
        if not eager:
            kwargs["compilation_config"] = {
                "level": 2,
                "cudagraph_mode": "FULL",
                "cudagraph_capture_sizes": sorted(
                    {b for b in range(1, batch + 1)}),
            }
    return LLM(**kwargs)


def load_prompts(dataset: str, n: int, tok, thinking: str):
    if dataset == "aime25":
        ds = load_dataset("MathArena/aime_2025", split=f"train[:{n}]")
        problems = [r["problem"] for r in ds]
        user_msgs = [aime26_prompt(p) for p in problems]
    elif dataset == "aime26":
        # AIME 2024 problems (30 in total). Truncate to n.
        try:
            ds = load_dataset("Maxwell-Jia/AIME_2024",
                              split=f"train[:{n}]")
            problems = [r["Problem"] for r in ds]
        except Exception:
            ds = load_dataset("HuggingFaceH4/aime_2024",
                              split=f"train[:{n}]")
            problems = [r["problem"] for r in ds]
        user_msgs = [aime26_prompt(p) for p in problems]
    elif dataset == "mmlu_pro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split=f"test[:{n}]")
        user_msgs = [mmlu_pro_prompt(r["question"], r["options"]) for r in ds]
    elif dataset == "hmmt":
        import pandas as _pd
        df = _pd.read_parquet("/data/datasets/zpo/hmmt.parquet")
        # Each row's prompt is an array of {role, content} dicts; take
        # the user message's content as the prompt body.
        user_msgs = [row["prompt"][0]["content"]
                     for _, row in df.iloc[:n].iterrows()]
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    prompts = []
    for m in user_msgs:
        chat = [{"role": "user", "content": m}]
        tmpl_kwargs = dict(tokenize=False, add_generation_prompt=True)
        # Try the model's `enable_thinking` knob first (Qwen3-style); fall
        # back to plain template if it's not supported.
        try:
            prompts.append(tok.apply_chat_template(
                chat, enable_thinking=(thinking == "on"), **tmpl_kwargs))
        except TypeError:
            prompts.append(tok.apply_chat_template(chat, **tmpl_kwargs))
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset",
                    choices=["mmlu_pro", "aime25", "aime26", "hmmt"],
                    default="mmlu_pro")
    ap.add_argument("--thinking",
                    choices=["on", "off"], default="on")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--explicit-captures", action="store_true")
    ap.add_argument("--mode", choices=["tidar", "ar"], default="tidar")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--n", type=int, default=16,
                    help="Number of prompts.")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--t-ar", type=float, default=0.05)
    ap.add_argument("--t-diff", type=float, default=0.0)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    prompts = load_prompts(args.dataset, args.n, tok, args.thinking)

    llm = build_llm(args.ckpt, args.mode, args.K, args.t_diff,
                    args.max_model_len, args.eager, args.gpu_mem,
                    batch=args.batch,
                    explicit_captures=args.explicit_captures)

    # Warmup so first-step compilation/capture isn't timed.
    _ = llm.generate(["hi"],
                     SamplingParams(temperature=0.0, max_tokens=4),
                     use_tqdm=False)

    sp = SamplingParams(n=1, temperature=args.t_ar,
                        max_tokens=args.max_tokens, seed=0)

    # Batched run -- all prompts at once, vLLM schedules up to
    # max_num_seqs (= args.batch) concurrently.
    t_all0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    total_elapsed = time.perf_counter() - t_all0

    per_prompt: list[dict] = []
    for i, o in enumerate(outs):
        comp = o.outputs[0]
        n_out = len(comp.token_ids)
        per_prompt.append({
            "idx": i, "n_out": n_out,
            "finish": comp.finish_reason,
        })
        print(f"[{args.tag}] prompt {i:2d}  n_out={n_out:4d}  "
              f"finish={comp.finish_reason}")

    total_tokens = sum(p["n_out"] for p in per_prompt)
    agg_tok_s = total_tokens / total_elapsed if total_elapsed > 0 else 0.0
    avg_tok_s_uw = sum(p["n_out"] / total_elapsed
                       for p in per_prompt) / max(1, len(per_prompt))

    print()
    print(f"=== {args.tag}  mode={args.mode}  BATCH={args.batch}  "
          f"P_env={os.environ.get('VLLM_TIDAR_PROPOSAL_ACC_LEVELS', '')}  "
          f"SF_TRITON={os.environ.get('VLLM_TIDAR_SF_TRITON', '0')}  "
          f"eager={args.eager} ===")
    print(f"[{args.tag}] BATCH={args.batch}  "
          f"total_prompts={len(per_prompt)}  "
          f"elapsed={total_elapsed:.2f}s  total_out={total_tokens}")
    print(f"total_tokens={total_tokens}  "
          f"total_elapsed={total_elapsed:.2f}s")
    print(f"agg_tok/s (sum_tok / sum_time)    = {agg_tok_s:7.2f}")
    print(f"avg_tok/s (unweighted per-prompt) = {avg_tok_s_uw:7.2f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "tag": args.tag, "mode": args.mode,
                "dataset": args.dataset, "thinking": args.thinking,
                "batch": args.batch,
                "P_env": os.environ.get(
                    "VLLM_TIDAR_PROPOSAL_ACC_LEVELS", ""),
                "SF_TRITON": os.environ.get("VLLM_TIDAR_SF_TRITON", "0"),
                "eager": args.eager,
                "K": args.K, "n": args.n,
                "max_tokens": args.max_tokens, "t_ar": args.t_ar,
                "total_tokens": total_tokens,
                "total_elapsed": total_elapsed,
                "agg_tok_per_s": agg_tok_s,
                "avg_tok_per_s_unweighted": avg_tok_s_uw,
                "per_prompt": per_prompt,
            }, f, indent=2)


if __name__ == "__main__":
    main()
