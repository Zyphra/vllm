# SPDX-License-Identifier: Apache-2.0
"""Probe V2 TiDAR two-forward throughput/acceptance on NVIDIA.

This is intentionally a thin offline LLM harness: one process, one GPU, one
batch size per invocation. It monkeypatches the V2 TiDAR speculator to count
accepted token lengths without synchronizing the host each decode step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.v1.worker.gpu import model_runner as v2_model_runner
from vllm.v1.worker.gpu.spec_decode import tidar as tidar_v2


DEFAULT_CKPT = "/data/checkpoints/smoediffusion_128k_64node-hf/iter_0012600"
DEFAULT_DATASET = "/data/home/jinzhao/aime26_thinkon.json"


class AcceptStats:
    def __init__(self) -> None:
        self.propose_calls = 0
        self.count_errors = 0
        self.max_active_reqs = 0
        self._device: torch.device | None = None
        self._sampled_sum: torch.Tensor | None = None
        self._active_reqs: torch.Tensor | None = None
        self._zero_sampled: torch.Tensor | None = None

    def reset(self) -> None:
        self.propose_calls = 0
        self.count_errors = 0
        self.max_active_reqs = 0
        if self._sampled_sum is not None:
            with torch.inference_mode():
                self._sampled_sum.zero_()
                self._active_reqs.zero_()
                self._zero_sampled.zero_()

    def _ensure_device(self, device: torch.device) -> None:
        if self._device == device and self._sampled_sum is not None:
            return
        self._device = device
        self._sampled_sum = torch.zeros((), dtype=torch.int64, device=device)
        self._active_reqs = torch.zeros((), dtype=torch.int64, device=device)
        self._zero_sampled = torch.zeros((), dtype=torch.int64, device=device)

    def observe(self, num_sampled: torch.Tensor, active_reqs: int) -> None:
        self.propose_calls += 1
        self.max_active_reqs = max(self.max_active_reqs, active_reqs)
        self._ensure_device(num_sampled.device)
        assert self._sampled_sum is not None
        assert self._active_reqs is not None
        assert self._zero_sampled is not None

        vals = num_sampled[:active_reqs].to(torch.int64)
        active = vals > 0
        self._sampled_sum.add_(vals.masked_select(active).sum())
        self._active_reqs.add_(active.sum())
        self._zero_sampled.add_((~active).sum())

    def snapshot(self) -> dict[str, int | float | None]:
        if self._sampled_sum is None or self._active_reqs is None:
            sampled_sum = 0
            active_reqs = 0
            zero_sampled = 0
        else:
            sampled_sum = int(self._sampled_sum.item())
            active_reqs = int(self._active_reqs.item())
            zero_sampled = int(self._zero_sampled.item())
        mean_accept = sampled_sum / active_reqs if active_reqs else None
        return {
            "propose_calls": self.propose_calls,
            "active_reqs": active_reqs,
            "sampled_sum": sampled_sum,
            "zero_sampled": zero_sampled,
            "max_active_reqs": self.max_active_reqs,
            "count_errors": self.count_errors,
            "mean_accept_len": mean_accept,
        }


ACCEPT_STATS = AcceptStats()


class ProfileStats:
    def __init__(self) -> None:
        self.enabled = os.environ.get("PATCH_PROBE_PROFILE", "0") == "1"
        self.max_events = int(os.environ.get("PATCH_PROBE_PROFILE_MAX_EVENTS",
                                             "20000"))
        self.reset()

    def reset(self) -> None:
        self.calls: Counter[str] = Counter()
        self.host_ns: Counter[str] = Counter()
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event,
                                          str]]] = defaultdict(list)
        self.shapes: dict[str, Counter[str]] = defaultdict(Counter)
        self.errors: Counter[str] = Counter()

    def _shape_key(self, meta: dict[str, Any] | None) -> str:
        if not meta:
            return "-"
        return ",".join(f"{k}={v}" for k, v in sorted(meta.items()))

    def timed_call(self, name: str, fn: Any, *args: Any,
                   meta: dict[str, Any] | None = None,
                   **kwargs: Any) -> Any:
        if not self.enabled:
            return fn(*args, **kwargs)
        self.calls[name] += 1
        shape_key = self._shape_key(meta)
        self.shapes[name][shape_key] += 1
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        host_start = time.perf_counter_ns()
        start_event.record()
        try:
            return fn(*args, **kwargs)
        except Exception:
            self.errors[name] += 1
            raise
        finally:
            end_event.record()
            self.host_ns[name] += time.perf_counter_ns() - host_start
            if len(self.events[name]) < self.max_events:
                self.events[name].append((start_event, end_event, shape_key))

    def snapshot(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        torch.cuda.synchronize()
        out: dict[str, Any] = {}
        for name in sorted(self.calls):
            event_pairs = self.events.get(name, [])
            shape_device_ms: Counter[str] = Counter()
            for start, end, shape_key in event_pairs:
                shape_device_ms[shape_key] += start.elapsed_time(end)
            device_ms = sum(shape_device_ms.values())
            calls = self.calls[name]
            host_ms = self.host_ns[name] / 1e6
            top_shape_timing = []
            for shape_key, count in self.shapes[name].most_common(8):
                shape_ms = float(shape_device_ms.get(shape_key, 0.0))
                recorded = sum(1 for _, _, key in event_pairs if key == shape_key)
                top_shape_timing.append({
                    "shape": shape_key,
                    "calls": count,
                    "events_recorded": recorded,
                    "device_ms_total": shape_ms,
                    "device_ms_mean": shape_ms / recorded if recorded else 0.0,
                })
            out[name] = {
                "calls": calls,
                "events_recorded": len(event_pairs),
                "host_ms_total": host_ms,
                "host_ms_mean": host_ms / calls if calls else 0.0,
                "device_ms_total": device_ms,
                "device_ms_mean": device_ms / len(event_pairs)
                if event_pairs else 0.0,
                "errors": self.errors[name],
                "top_shapes": self.shapes[name].most_common(8),
                "top_shape_timing": top_shape_timing,
            }
        return out


PROFILE_STATS = ProfileStats()
_ORIG_PROPOSE = tidar_v2.TiDARSpeculator.propose
_ORIG_TIDAR_RUN_MODEL = tidar_v2.TiDARSpeculator.run_model
_ORIG_EXECUTE_MODEL = v2_model_runner.GPUModelRunner.execute_model
_ORIG_SAMPLE = v2_model_runner.GPUModelRunner.sample
_ORIG_SAMPLE_TOKENS = v2_model_runner.GPUModelRunner.sample_tokens
_ORIG_PROPOSE_DRAFT = v2_model_runner.GPUModelRunner.propose_draft


def _counted_propose(self: tidar_v2.TiDARSpeculator, *args: Any,
                     **kwargs: Any) -> torch.Tensor:
    try:
        input_batch = args[0]
        num_sampled = args[3]
        ACCEPT_STATS.observe(num_sampled, int(input_batch.num_reqs))
    except Exception:
        ACCEPT_STATS.count_errors += 1
    meta = None
    if PROFILE_STATS.enabled:
        try:
            input_batch = args[0]
            meta = {
                "reqs": int(input_batch.num_reqs),
                "tokens": int(input_batch.num_reqs)
                * (int(self.num_speculative_steps) + 1),
            }
        except Exception:
            meta = None
    return PROFILE_STATS.timed_call(
        "tidar_propose_total", _ORIG_PROPOSE, self, *args, meta=meta, **kwargs)


tidar_v2.TiDARSpeculator.propose = _counted_propose


def _profiled_tidar_run_model(self: tidar_v2.TiDARSpeculator, *args: Any,
                              **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    meta = None
    if PROFILE_STATS.enabled:
        meta = {
            "tokens": int(args[0]) if args else int(kwargs.get("num_tokens", 0)),
            "cg": kwargs.get("cudagraph_size") is not None,
        }
    return PROFILE_STATS.timed_call(
        "tidar_draft_run_model", _ORIG_TIDAR_RUN_MODEL, self, *args,
        meta=meta, **kwargs)


tidar_v2.TiDARSpeculator.run_model = _profiled_tidar_run_model


def _profiled_execute_model(self: v2_model_runner.GPUModelRunner,
                            scheduler_output: Any, *args: Any,
                            **kwargs: Any) -> Any:
    meta = None
    if PROFILE_STATS.enabled:
        meta = {
            "reqs": len(scheduler_output.num_scheduled_tokens),
            "tokens": int(scheduler_output.total_num_scheduled_tokens),
            "spec": bool(scheduler_output.scheduled_spec_decode_tokens),
        }
    return PROFILE_STATS.timed_call(
        "target_execute_model", _ORIG_EXECUTE_MODEL, self, scheduler_output,
        *args, meta=meta, **kwargs)


def _profiled_sample(self: v2_model_runner.GPUModelRunner, hidden_states: torch.Tensor,
                     input_batch: Any, grammar_output: Any) -> Any:
    meta = None
    if PROFILE_STATS.enabled:
        meta = {
            "reqs": int(input_batch.num_reqs),
            "tokens": int(input_batch.num_tokens),
            "draft_tokens": int(input_batch.num_draft_tokens),
        }
    return PROFILE_STATS.timed_call(
        "sample_logits_sampler_reject", _ORIG_SAMPLE, self, hidden_states,
        input_batch, grammar_output, meta=meta)


def _profiled_sample_tokens(self: v2_model_runner.GPUModelRunner,
                            grammar_output: Any) -> Any:
    return PROFILE_STATS.timed_call(
        "sample_tokens_total", _ORIG_SAMPLE_TOKENS, self, grammar_output)


def _profiled_propose_draft(self: v2_model_runner.GPUModelRunner,
                            input_batch: Any, *args: Any,
                            **kwargs: Any) -> torch.Tensor:
    meta = None
    if PROFILE_STATS.enabled:
        meta = {
            "reqs": int(input_batch.num_reqs),
            "tokens": int(input_batch.num_reqs)
            * (int(self.num_speculative_steps) + 1),
        }
    return PROFILE_STATS.timed_call(
        "propose_draft_wrapper", _ORIG_PROPOSE_DRAFT, self, input_batch,
        *args, meta=meta, **kwargs)


v2_model_runner.GPUModelRunner.execute_model = _profiled_execute_model
v2_model_runner.GPUModelRunner.sample = _profiled_sample
v2_model_runner.GPUModelRunner.sample_tokens = _profiled_sample_tokens
v2_model_runner.GPUModelRunner.propose_draft = _profiled_propose_draft


def load_prompt_texts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        if "prompts" in raw:
            raw = raw["prompts"]
        elif "data" in raw:
            raw = raw["data"]
        else:
            raw = list(raw.values())
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


def select_prompts(prompts: list[str], batch: int, offset: int) -> list[str]:
    return [prompts[(offset + i) % len(prompts)] for i in range(batch)]


def encode_prompts(
    prompts: list[str],
    checkpoint: str,
) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    prompt_inputs = [
        {"prompt_token_ids": tokenizer.encode(p, add_special_tokens=False)}
        for p in prompts
    ]
    first_ids = prompt_inputs[0]["prompt_token_ids"] if prompt_inputs else []
    bos_id = tokenizer.bos_token_id
    leading_bos_count = sum(1 for token_id in first_ids[:4] if token_id == bos_id)
    return prompt_inputs, {
        "bos_token_id": bos_id,
        "leading4": first_ids[:4],
        "leading_bos_count": leading_bos_count,
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
    idx = min(len(sorted_lens) - 1, int(round((len(sorted_lens) - 1) * pct)))
    return int(sorted_lens[idx])


def run_once(llm: LLM, prompts: list[str], sampling: SamplingParams) -> dict[str, Any]:
    ACCEPT_STATS.reset()
    PROFILE_STATS.reset()
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_tokens, lens = output_token_count(outputs)
    lens_sorted = sorted(lens)
    stats = ACCEPT_STATS.snapshot()
    legacy_mean = (
        total_tokens / (len(prompts) * stats["propose_calls"])
        if stats["propose_calls"] else None
    )
    return {
        **stats,
        "elapsed_s": elapsed,
        "total_tokens": total_tokens,
        "throughput_tok_s": total_tokens / elapsed,
        "legacy_mean_accept_len": legacy_mean,
        "len_min": min(lens) if lens else 0,
        "len_p50": percentile(lens_sorted, 0.50),
        "len_p90": percentile(lens_sorted, 0.90),
        "len_max": max(lens) if lens else 0,
        "token_hash": token_hash(outputs),
        "profile": PROFILE_STATS.snapshot(),
        "sample_text": outputs[0].outputs[0].text[:240] if outputs else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.environ.get("PATCH_PROBE_CKPT",
                                                        DEFAULT_CKPT))
    parser.add_argument("--dataset", default=os.environ.get(
        "PATCH_PROBE_DATASET", DEFAULT_DATASET))
    parser.add_argument("--batch", type=int, default=int(os.environ.get(
        "PATCH_PROBE_BATCH", "16")))
    parser.add_argument("--num-prompts", type=int, default=int(os.environ.get(
        "PATCH_PROBE_NUM_PROMPTS", "0")),
        help="Number of dataset prompts to select before n-sample replication. "
        "Defaults to --batch for backward compatibility.")
    parser.add_argument("--max-num-seqs", type=int, default=int(os.environ.get(
        "PATCH_PROBE_MAX_NUM_SEQS", "0")),
        help="LLM max_num_seqs. Defaults to --batch for backward compatibility.")
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get(
        "PATCH_PROBE_MAX_TOKENS", "128")))
    parser.add_argument("--warmup-tokens", type=int, default=int(os.environ.get(
        "PATCH_PROBE_WARMUP_TOKENS", "32")))
    parser.add_argument("--repeats", type=int, default=int(os.environ.get(
        "PATCH_PROBE_REPEATS", "2")))
    parser.add_argument("--offset", type=int, default=int(os.environ.get(
        "PATCH_PROBE_OFFSET", "0")))
    parser.add_argument("--n-sample", type=int, default=int(os.environ.get(
        "PATCH_PROBE_NSAMPLE", "1")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get(
        "PATCH_PROBE_SEED", "0")))
    parser.add_argument("--target-temp", type=float, default=float(
        os.environ.get("PATCH_PROBE_TARGET_TEMP", "0.0")))
    parser.add_argument("--draft-temp", type=float, default=float(
        os.environ.get("PATCH_PROBE_DRAFT_TEMP", "0.0")))
    parser.add_argument("--num-spec-tokens", type=int, default=int(
        os.environ.get("PATCH_PROBE_NUM_SPEC_TOKENS", "16")))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get(
        "PATCH_PROBE_MAX_MODEL_LEN", "10000")))
    parser.add_argument("--max-num-batched-tokens", type=int, default=int(
        os.environ.get("PATCH_PROBE_MAX_NUM_BATCHED_TOKENS", "8192")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(
        os.environ.get("PATCH_PROBE_GPU_MEM_UTIL", "0.85")))
    parser.add_argument("--backend", default=os.environ.get(
        "PATCH_PROBE_BACKEND", "FLASH_ATTN"))
    parser.add_argument("--cudagraph-mode", default=os.environ.get(
        "PATCH_PROBE_CGMODE", "FULL_AND_PIECEWISE"))
    parser.add_argument("--enforce-eager", action="store_true",
                        default=os.environ.get("PATCH_PROBE_EAGER") == "1")
    parser.add_argument("--enable-log-stats", action="store_true",
                        default=os.environ.get(
                            "PATCH_PROBE_ENABLE_LOG_STATS") == "1")
    parser.add_argument("--prompt-token-ids", action="store_true",
                        default=os.environ.get("PATCH_PROBE_PROMPT_TOKEN_IDS") == "1")
    parser.add_argument("--ignore-eos", action="store_true",
                        default=os.environ.get("PATCH_PROBE_IGNORE_EOS") == "1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_step_tokens = args.num_spec_tokens + 1
    num_prompts = args.num_prompts or args.batch
    max_num_seqs = args.max_num_seqs or args.batch
    max_num_batched_tokens = max(
        args.max_num_batched_tokens, max_num_seqs * per_step_tokens)
    all_prompts = load_prompt_texts(args.dataset)
    base_prompts = select_prompts(all_prompts, num_prompts, args.offset)
    selected_prompts = base_prompts * args.n_sample
    if args.prompt_token_ids:
        prompts, tokenization = encode_prompts(selected_prompts, args.ckpt)
    else:
        prompts = selected_prompts
        tokenization = None

    print("PATCH_PROBE_CONTEXT " + json.dumps({
        "ckpt": args.ckpt,
        "dataset": args.dataset,
        "batch": args.batch,
        "num_prompts": num_prompts,
        "max_num_seqs": max_num_seqs,
        "n_sample": args.n_sample,
        "num_input_sequences": len(prompts),
        "num_dataset_prompts": len(all_prompts),
        "prompt_token_ids": args.prompt_token_ids,
        "tokenization": tokenization,
        "max_tokens": args.max_tokens,
        "ignore_eos": args.ignore_eos,
        "warmup_tokens": args.warmup_tokens,
        "repeats": args.repeats,
        "offset": args.offset,
        "seed": args.seed,
        "target_temp": args.target_temp,
        "draft_temp": args.draft_temp,
        "num_spec_tokens": args.num_spec_tokens,
        "backend": args.backend,
        "cudagraph_mode": args.cudagraph_mode,
        "enforce_eager": args.enforce_eager,
        "enable_log_stats": args.enable_log_stats,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "vllm_use_v2_model_runner": os.environ.get("VLLM_USE_V2_MODEL_RUNNER"),
        "vllm_tidar_two_forward": os.environ.get("VLLM_TIDAR_TWO_FORWARD"),
        "vllm_flash_attn_version": os.environ.get("VLLM_FLASH_ATTN_VERSION"),
    }, sort_keys=True), flush=True)

    llm_kwargs: dict[str, Any] = {
        "model": args.ckpt,
        "dtype": "bfloat16",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
        "swap_space": 4.0,
        "attention_backend": args.backend,
        "async_scheduling": True,
        "distributed_executor_backend": "uni",
        "disable_log_stats": not args.enable_log_stats,
        "speculative_config": {
            "method": "tidar",
            "num_speculative_tokens": args.num_spec_tokens,
            "tidar_diff_temperature": args.draft_temp,
        },
    }
    if not args.enforce_eager:
        llm_kwargs["compilation_config"] = {
            "cudagraph_mode": args.cudagraph_mode,
        }

    print("PATCH_PROBE_LOAD_START", flush=True)
    llm = LLM(**llm_kwargs)
    print("PATCH_PROBE_LOAD_DONE", flush=True)

    warmup_sampling = SamplingParams(
        n=1, temperature=args.target_temp, max_tokens=args.warmup_tokens,
        seed=args.seed, ignore_eos=args.ignore_eos)
    print("PATCH_PROBE_WARMUP_START", flush=True)
    _ = llm.generate(prompts, warmup_sampling, use_tqdm=False)
    torch.cuda.synchronize()
    print("PATCH_PROBE_WARMUP_DONE", flush=True)

    sampling = SamplingParams(
        n=1, temperature=args.target_temp, max_tokens=args.max_tokens,
        seed=args.seed, ignore_eos=args.ignore_eos)
    results = []
    for repeat_idx in range(args.repeats):
        result = run_once(llm, prompts, sampling)
        result.update({
            "repeat": repeat_idx,
            "batch": args.batch,
            "num_prompts": num_prompts,
            "max_num_seqs": max_num_seqs,
            "num_input_sequences": len(prompts),
            "max_tokens": args.max_tokens,
            "target_temp": args.target_temp,
            "draft_temp": args.draft_temp,
            "backend": args.backend,
            "cudagraph_mode": args.cudagraph_mode,
        })
        results.append(result)
        print("PATCH_PROBE_RESULT " + json.dumps(result, sort_keys=True),
              flush=True)

    if results:
        best = max(results, key=lambda x: x["throughput_tok_s"])
        print("PATCH_PROBE_BEST " + json.dumps(best, sort_keys=True),
              flush=True)


if __name__ == "__main__":
    main()
