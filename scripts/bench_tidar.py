"""Direct TiDAR throughput benchmark matching handoff.md table.

AIME25 n=30, thinking-off, K=16, max_tokens=10000, T_AR=0.
Configurable batch + proposal_acc_levels via env.
"""
import os
import sys
import time

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset


CKPT = "/data/checkpoints/smoediffusion_128k_64node-hf/iter_0012000"


def aime25_prompt(p: str) -> str:
    return p


def build_llm(K: int, batch: int, mode: str,
              gpu_mem: float, enforce_eager: bool,
              cudagraph_mode: str,
              max_model_len: int = 4096):
    log_stats_disabled = (
        os.environ.get("BENCH_DISABLE_LOG_STATS", "0") == "1")
    kwargs = dict(
        model=CKPT, dtype="bfloat16",
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        max_num_seqs=batch,
        enforce_eager=enforce_eager,
        seed=0,
        disable_log_stats=log_stats_disabled,
        swap_space=16.0,
        attention_backend=os.environ.get("VLLM_ATTENTION_BACKEND", "FLEX_ATTENTION"),
    )
    _mnbt = os.environ.get("BENCH_MNBT")
    if _mnbt:
        kwargs["max_num_batched_tokens"] = int(_mnbt)
    if os.environ.get("BENCH_FI_AUTOTUNE_OFF", "0") == "1":
        kwargs["kernel_config"] = {"enable_flashinfer_autotune": False}
    _olvl = os.environ.get("BENCH_O")
    if _olvl:
        kwargs["optimization_level"] = int(_olvl)
    if mode in ("tf", "sf"):
        kwargs["speculative_config"] = {
            "method": "tidar",
            "num_speculative_tokens": K,
            "tidar_diff_temperature": 0.0,
        }
    if not enforce_eager:
        cc = {
            "cudagraph_mode": cudagraph_mode,
            "cudagraph_copy_inputs": True,
        }
        if os.environ.get("BENCH_COMBO_OFF", "0") == "1":
            cc["inductor_compile_config"] = {
                "combo_kernels": False,
                "benchmark_combo_kernel": False,
            }
        # When cg_mode=NONE, seed compile_sizes so warmup does not try
        # compile_range.end (=MNBT+1) and hit the size assert.
        if cudagraph_mode == "NONE":
            cc["compile_sizes"] = [1]
        _cmode = os.environ.get("BENCH_COMPILE_MODE")
        if _cmode:
            cc["mode"] = _cmode  # NONE, STOCK_TORCH_COMPILE, DYNAMO_TRACE_ONCE, VLLM_COMPILE
        kwargs["compilation_config"] = cc
    return LLM(**kwargs)


def main():
    K = int(os.environ.get("BENCH_K", "16"))
    batch = int(os.environ.get("BENCH_B", "1"))
    n = int(os.environ.get("BENCH_N", "30"))
    max_tokens = int(os.environ.get("BENCH_MT", "10000"))
    mode = os.environ.get("BENCH_MODE", "sf")  # ar | tf | sf
    enforce_eager = os.environ.get("BENCH_EAGER", "0") == "1"
    cg_mode = os.environ.get("BENCH_CG", "FULL_DECODE_ONLY")
    gpu_mem = float(os.environ.get("BENCH_GPU_MEM", "0.5"))
    max_model_len = int(os.environ.get("BENCH_MML", "4096"))

    if mode == "tf":
        os.environ["VLLM_TIDAR_TWO_FORWARD"] = "1"

    tok = AutoTokenizer.from_pretrained(CKPT)
    ds = load_dataset("MathArena/aime_2025", split=f"train[:{n}]")
    problems = [r["problem"] for r in ds]
    chats = [[{"role": "user", "content": p}] for p in problems]
    prompts = [
        tok.apply_chat_template(c, tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=False)
        for c in chats
    ]

    llm = build_llm(K, batch, mode, gpu_mem, enforce_eager, cg_mode,
                    max_model_len)

    sp = SamplingParams(n=1, temperature=0.0, max_tokens=max_tokens, seed=0)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    tok_per_s = total_tokens / dt

    print()
    print(f"=== BENCH mode={mode} K={K} batch={batch} n={n} "
          f"max_tok={max_tokens} eager={enforce_eager} cg={cg_mode} "
          f"P_env={os.environ.get('VLLM_TIDAR_PROPOSAL_ACC_LEVELS', '')} ===")
    print(f"total_tokens={total_tokens}  elapsed={dt:.2f}s")
    print(f"agg_tok/s = {tok_per_s:7.2f}")
    for i, o in enumerate(outs[:3]):
        toks = o.outputs[0].token_ids
        print(f"  prompt {i}: n_out={len(toks)} finish={o.outputs[0].finish_reason}")
        print(f"    last 120ch: {repr(tok.decode(toks)[-120:])}")


if __name__ == "__main__":
    main()
