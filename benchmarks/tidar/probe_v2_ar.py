# SPDX-License-Identifier: Apache-2.0
"""Probe V2 AR throughput with the same workload as the TiDAR TF tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from collections.abc import Sequence
from typing import Any

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.v1.worker.gpu import model_runner as v2_model_runner


class SteadyStateStats:
    """Measure complete full-batch AR decode iterations on the GPU."""

    def __init__(self) -> None:
        self.enabled = os.environ.get("PATCH_PROBE_STEADY", "0") == "1"
        self.batch = int(os.environ.get("PATCH_PROBE_STEADY_BATCH", "0"))
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self._events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def begin(self, scheduler_output: Any) -> torch.cuda.Event | None:
        if not self.enabled or self.batch <= 0:
            return None
        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_tokens = int(scheduler_output.total_num_scheduled_tokens)
        is_spec = bool(scheduler_output.scheduled_spec_decode_tokens)
        if num_reqs != self.batch or num_tokens != self.batch or is_spec:
            return None
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        return start

    def finish(self, start: torch.cuda.Event | None) -> None:
        if start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._events.append((start, end))
        self.calls += 1

    def snapshot(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        torch.cuda.synchronize()
        device_ms = sum(start.elapsed_time(end) for start, end in self._events)
        tokens = self.calls * self.batch
        return {
            "batch": self.batch,
            "calls": self.calls,
            "tokens": tokens,
            "device_ms_total": device_ms,
            "device_ms_mean": device_ms / self.calls if self.calls else None,
            "device_throughput_tok_s": (
                tokens / (device_ms / 1000) if device_ms else None),
        }


STEADY_STATS = SteadyStateStats()
_ORIG_EXECUTE_MODEL = v2_model_runner.GPUModelRunner.execute_model


def _steady_execute_model(
    self: v2_model_runner.GPUModelRunner,
    scheduler_output: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    start = STEADY_STATS.begin(scheduler_output)
    try:
        return _ORIG_EXECUTE_MODEL(self, scheduler_output, *args, **kwargs)
    finally:
        STEADY_STATS.finish(start)


v2_model_runner.GPUModelRunner.execute_model = _steady_execute_model


def load_prompt_texts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("prompts") or raw.get("data") or list(raw.values())
    prompts: list[str] = []
    for item in raw:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict):
            for key in ("prompt", "text", "input", "question"):
                if isinstance(item.get(key), str):
                    prompts.append(item[key])
                    break
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def encode_prompts(
    prompts: list[str],
    checkpoint: str,
    force_bos: bool,
    prompt_token_limit: int,
) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    encoded = []
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if force_bos:
            bos_id = tokenizer.bos_token_id
            if bos_id is None:
                raise ValueError("--force-bos requires a tokenizer BOS token")
            while token_ids and token_ids[0] == bos_id:
                token_ids.pop(0)
            token_ids.insert(0, bos_id)
        if prompt_token_limit > 0:
            token_ids = token_ids[:prompt_token_limit]
        encoded.append({"prompt_token_ids": token_ids})
    first_ids = encoded[0]["prompt_token_ids"] if encoded else []
    bos_id = tokenizer.bos_token_id
    return encoded, {
        "bos_token_id": bos_id,
        "leading4": first_ids[:4],
        "leading_bos_count": sum(
            1 for token_id in first_ids[:4] if token_id == bos_id),
    }


def token_hash(outputs: Sequence[Any]) -> str:
    h = hashlib.sha256()
    for request_output in outputs:
        for completion in request_output.outputs:
            h.update(bytes(str(completion.token_ids), "utf-8"))
            h.update(b"\n")
    return h.hexdigest()[:16]


def output_token_count(outputs: Sequence[Any]) -> tuple[int, list[int]]:
    lens = [
        len(completion.token_ids)
        for request_output in outputs
        for completion in request_output.outputs
    ]
    return sum(lens), lens


def percentile(sorted_lens: Sequence[int], pct: float) -> int:
    if not sorted_lens:
        return 0
    idx = min(len(sorted_lens) - 1,
              int(round((len(sorted_lens) - 1) * pct)))
    return int(sorted_lens[idx])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--n-sample", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--warmup-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-temp", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=12000)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--backend", default="ROCM_AITER_FA")
    parser.add_argument("--cudagraph-mode", default="FULL_AND_PIECEWISE")
    parser.add_argument("--prompt-token-ids", action="store_true")
    parser.add_argument("--force-bos", action="store_true",
                        help="Prepend exactly one BOS to prompt token IDs.")
    parser.add_argument("--prompt-token-limit", type=int, default=0)
    parser.add_argument("--ignore-eos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.force_bos and not args.prompt_token_ids:
        raise ValueError("--force-bos requires --prompt-token-ids")

    all_prompts = load_prompt_texts(args.dataset)
    base_prompts = [
        all_prompts[i % len(all_prompts)] for i in range(args.num_prompts)
    ]
    selected_prompts = base_prompts * args.n_sample
    tokenization = None
    if args.prompt_token_ids:
        prompts, tokenization = encode_prompts(
            selected_prompts, args.ckpt, args.force_bos,
            args.prompt_token_limit)
    else:
        prompts = selected_prompts

    print("PATCH_PROBE_CONTEXT " + json.dumps({
        "mode": "AR",
        "host": socket.gethostname(),
        "ckpt": args.ckpt,
        "dataset": args.dataset,
        "batch": args.batch,
        "num_prompts": args.num_prompts,
        "max_num_seqs": args.max_num_seqs,
        "n_sample": args.n_sample,
        "num_input_sequences": len(prompts),
        "num_dataset_prompts": len(all_prompts),
        "prompt_token_ids": args.prompt_token_ids,
        "force_bos": args.force_bos,
        "prompt_token_limit": args.prompt_token_limit,
        "tokenization": tokenization,
        "max_tokens": args.max_tokens,
        "ignore_eos": args.ignore_eos,
        "warmup_tokens": args.warmup_tokens,
        "repeats": args.repeats,
        "seed": args.seed,
        "target_temp": args.target_temp,
        "backend": args.backend,
        "cudagraph_mode": args.cudagraph_mode,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "vllm_use_v2_model_runner": os.environ.get(
            "VLLM_USE_V2_MODEL_RUNNER"),
    }, sort_keys=True), flush=True)

    llm = LLM(
        model=args.ckpt,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=False,
        seed=args.seed,
        swap_space=4.0,
        attention_backend=args.backend,
        async_scheduling=True,
        distributed_executor_backend="uni",
        disable_log_stats=True,
        compilation_config={"cudagraph_mode": args.cudagraph_mode},
    )

    warmup_sampling = SamplingParams(
        n=1,
        temperature=args.target_temp,
        max_tokens=args.warmup_tokens,
        seed=args.seed,
        ignore_eos=args.ignore_eos,
    )
    print("PATCH_PROBE_WARMUP_START", flush=True)
    _ = llm.generate(prompts, warmup_sampling, use_tqdm=False)
    torch.cuda.synchronize()
    print("PATCH_PROBE_WARMUP_DONE", flush=True)

    sampling = SamplingParams(
        n=1,
        temperature=args.target_temp,
        max_tokens=args.max_tokens,
        seed=args.seed,
        ignore_eos=args.ignore_eos,
    )
    results = []
    for repeat_idx in range(args.repeats):
        STEADY_STATS.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        total_tokens, lens = output_token_count(outputs)
        lens_sorted = sorted(lens)
        result = {
            "mode": "AR",
            "repeat": repeat_idx,
            "batch": args.batch,
            "num_prompts": args.num_prompts,
            "max_num_seqs": args.max_num_seqs,
            "num_input_sequences": len(prompts),
            "max_tokens": args.max_tokens,
            "target_temp": args.target_temp,
            "backend": args.backend,
            "cudagraph_mode": args.cudagraph_mode,
            "elapsed_s": elapsed,
            "total_tokens": total_tokens,
            "throughput_tok_s": total_tokens / elapsed,
            "len_min": min(lens) if lens else 0,
            "len_p50": percentile(lens_sorted, 0.50),
            "len_p90": percentile(lens_sorted, 0.90),
            "len_max": max(lens) if lens else 0,
            "token_hash": token_hash(outputs),
            "steady_state": STEADY_STATS.snapshot(),
            "sample_text": outputs[0].outputs[0].text[:240]
            if outputs else "",
        }
        results.append(result)
        print("PATCH_PROBE_RESULT " + json.dumps(result, sort_keys=True),
              flush=True)
    if results:
        best = max(results, key=lambda x: x["throughput_tok_s"])
        print("PATCH_PROBE_BEST " + json.dumps(best, sort_keys=True),
              flush=True)


if __name__ == "__main__":
    main()
