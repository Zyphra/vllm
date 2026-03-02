#!/usr/bin/env python3
"""
Async streaming eval script for vLLM endpoint.
All completions stream concurrently. Aggregate progress on interval.
NaN logprob and token_id=0 detection with noisy logging.

Supports loading from HuggingFace datasets or local parquet files.
Optional robust math scoring with an embedded verifier in a Pebble pool.
Reports mean@N and pass@k metrics.

Example usage:
# Start vLLM server with:
vllm serve --model=Zyphra-staging/smoe-brr_sft_v4_iter_0009900 \
    --served-model-name=smoe --port=8001 --api-key Op0mxhgL8rbKuX0fBDw \
    --seed=0 --dtype=bfloat16 --gpu-memory-utilization=0.9 \
    --max-num-seqs 128 --max-num-batched-tokens 65536 \
    --reasoning-parser deepseek_r1

# Run evaluation script with:
python eval_stream.py \
    --concurrency 1024 \
    --max-tokens 60000 -N 8 \
    --api-key Op0mxhgL8rbKuX0fBDw \
    --base-url http://localhost:8001 \
    --hf-dataset MathArena/aime_2025 \
    --hf-split train --score

"""

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
import pandas as pd

try:
    from pebble import ProcessPool  # type: ignore[import-not-found]
    try:
        from pebble.common import ProcessExpired  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover (older pebble)
        class ProcessExpired(Exception):  # type: ignore[no-redef]
            pass
except ImportError:  # pragma: no cover (runtime fallback)
    ProcessPool = None  # type: ignore[assignment]
    class ProcessExpired(Exception):  # type: ignore[no-redef]
        pass


# ── ANSI colors ──────────────────────────────────────────────────────────────
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


MATH_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."


def _to_native(obj):
    """Recursively convert numpy types to native Python."""
    if isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.str_):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return obj


def load_dataset_hf(dataset_path: str, split: str = "test", num_prompts: int | None = None) -> pd.DataFrame:
    """Load a HuggingFace dataset and return as DataFrame."""
    from datasets import load_dataset
    ds = load_dataset(dataset_path, split=split)
    df = ds.to_pandas()
    if num_prompts is not None:
        df = df.head(num_prompts)
    return df


def load_dataset_parquet(path: str, num_prompts: int | None = None) -> pd.DataFrame:
    """Load a local parquet file."""
    df = pd.read_parquet(path)
    if num_prompts is not None:
        df = df.head(num_prompts)
    return df


def _pool_initializer():
    """Keep each verifier subprocess single-threaded."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _strip_reasoning_trace(text: str) -> str:
    if text is None:
        return ""
    think_end = text.rfind("</think>")
    if think_end != -1:
        return text[think_end + len("</think>"):]
    return text


def _coerce_ground_truth(answer: Any) -> Any:
    """Keep answer shape when possible (supports list ground truths)."""
    if answer is None:
        return ""
    if isinstance(answer, (list, tuple, int, float)):
        return answer
    if isinstance(answer, str):
        stripped = answer.strip()
        # Keep compatibility with datasets that store list answers as JSON strings.
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return answer
    return str(answer)


def _to_answer_list(answer: Any) -> list[str]:
    if isinstance(answer, (list, tuple)):
        return [str(x) for x in answer]
    return [str(answer)]


def _extract_last_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    lb = text.find("{", idx)
    if lb < 0:
        return None
    depth = 0
    rb = -1
    for i in range(lb, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                rb = i
                break
    if rb < 0:
        return None
    return text[lb + 1 : rb]


def _normalize_simple_answer(s: str) -> str:
    s = str(s).strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("$", "")
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".,;:")
    return s.lower()


def _score_with_simple_match(completion_text: str, answer: Any) -> int:
    """Last-resort fallback when full verifiers are unavailable."""
    pred_visible = _strip_reasoning_trace(completion_text)
    pred_candidates = [pred_visible]
    pred_boxed = _extract_last_boxed(pred_visible)
    if pred_boxed:
        pred_candidates.append(pred_boxed)

    gt_candidates: list[str] = []
    for gt in _to_answer_list(answer):
        gt_candidates.append(gt)
        gt_boxed = _extract_last_boxed(gt)
        if gt_boxed:
            gt_candidates.append(gt_boxed)

    pred_norm = {_normalize_simple_answer(x) for x in pred_candidates if str(x).strip()}
    gt_norm = {_normalize_simple_answer(x) for x in gt_candidates if str(x).strip()}
    if not pred_norm or not gt_norm:
        return 0
    return 1 if (pred_norm & gt_norm) else 0


def _strip_string_mathd(string: str) -> str:
    def _fix_fracs(s: str) -> str:
        substrs = s.split("\\frac")
        new_str = substrs[0]
        if len(substrs) <= 1:
            return s
        for substr in substrs[1:]:
            new_str += "\\frac"
            if not substr:
                return s
            if substr[0] == "{":
                new_str += substr
                continue
            if len(substr) < 2:
                return s
            a = substr[0]
            b = substr[1]
            if b != "{":
                new_str += "{" + a + "}{" + b + "}" + substr[2:]
            else:
                new_str += "{" + a + "}" + substr[1:]
        return new_str

    def _fix_a_slash_b(s: str) -> str:
        if len(s.split("/")) != 2:
            return s
        a, b = s.split("/")
        try:
            a_int = int(a)
            b_int = int(b)
            if s == f"{a_int}/{b_int}":
                return f"\\frac{{{a_int}}}{{{b_int}}}"
        except Exception:
            return s
        return s

    def _fix_sqrt(s: str) -> str:
        if "\\sqrt" not in s:
            return s
        out = s.split("\\sqrt")[0]
        for split in s.split("\\sqrt")[1:]:
            if not split:
                continue
            if split[0] != "{":
                out += "\\sqrt{" + split[0] + "}" + split[1:]
            else:
                out += "\\sqrt" + split
        return out

    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("\\%", "").replace("\%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if string and string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string).replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    return _fix_a_slash_b(string)


def _mathd_normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
    if m is not None:
        answer = m.group("text").strip()
    try:
        return _strip_string_mathd(answer)
    except Exception:
        return answer


def _grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    return _mathd_normalize_answer(given_answer) == _mathd_normalize_answer(ground_truth)


def _normalize_sympy_expr(expr: str | None) -> str | None:
    if expr is None:
        return None
    expr = str(expr).strip()
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")
    expr = expr.replace("\\%", "%").replace("\\$", "$").replace("$", "").replace("%", "")
    expr = expr.replace(" or ", " , ").replace(" and ", " , ")
    expr = expr.replace("million", "*10^6").replace("billion", "*10^9").replace("trillion", "*10^12")
    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)
    expr = re.sub(r",\\! *", "", expr)
    expr = re.sub(r"- *", "-", expr)
    expr = expr.replace("{", "").replace("}", "")
    expr = re.sub(r"([0-9]) +([0-9])", r"\1+\2", expr)
    expr = expr.replace(" ", "")
    return expr.lower()


def _sympy_equiv(a: str, b: str) -> bool:
    try:
        import sympy
        from sympy.parsing import sympy_parser

        expr = f"({a})-({b})".replace("^", "**")
        parsed = sympy_parser.parse_expr(
            expr,
            transformations=(
                sympy_parser.standard_transformations
                + (sympy_parser.implicit_multiplication_application,)
            ),
        )
        return bool(sympy.simplify(parsed) == 0)
    except Exception:
        return False


def _grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    gt = _normalize_sympy_expr(ground_truth)
    pred = _normalize_sympy_expr(given_answer)
    if gt is None or pred is None:
        return False
    if gt == pred:
        return True
    if not pred:
        return False
    return _sympy_equiv(gt, pred)


def _score_with_boxed_equivalence(completion_text: str, answer: Any) -> int:
    pred_boxed = _extract_last_boxed(completion_text)
    if pred_boxed is None:
        return 0

    ground_truths = _to_answer_list(answer)
    processed_truths = []
    for truth in ground_truths:
        if "\\boxed" in truth:
            unboxed = _extract_last_boxed(truth)
            if unboxed is not None:
                processed_truths.append(unboxed)
        else:
            processed_truths.append(truth)

    if not processed_truths:
        return 0

    for gt in processed_truths:
        if _grade_answer_mathd(pred_boxed, gt) or _grade_answer_sympy(pred_boxed, gt):
            return 1
    return 0


def _score_with_math_verify_fallback(completion_text: str, answer: Any) -> int:
    from math_verify import parse, verify  # type: ignore[import-not-found]
    from math_verify.parser import LatexExtractionConfig  # type: ignore[import-not-found]

    pred = parse(completion_text, extraction_config=[LatexExtractionConfig()])
    for gt in _to_answer_list(answer):
        try:
            gold = parse(gt)
        except Exception:
            continue
        if verify(gold, pred):
            return 1
    return 0


def _run_verifier_preflight() -> tuple[bool, str]:
    """Report which embedded verifier backends are available."""
    available = []
    missing = []
    try:
        import sympy  # noqa: F401

        available.append("sympy")
    except Exception as e:
        missing.append(f"sympy={type(e).__name__}")

    try:
        import math_verify  # type: ignore[import-not-found]  # noqa: F401

        available.append("math_verify")
    except Exception as e:
        missing.append(f"math_verify={type(e).__name__}")

    available.append("simple_match")
    ok = any(name in {"sympy", "math_verify"} for name in available)
    msg = f"embedded backends: {', '.join(available)}"
    if missing:
        msg += f" (missing: {', '.join(missing)})"
    return ok, msg


def _score_with_archer(payload: tuple[str, Any]) -> int:
    """Worker function for Pebble pool. Returns 1 for correct else 0."""
    completion_text, answer = payload
    completion_text = _strip_reasoning_trace(completion_text)
    answer = _coerce_ground_truth(answer)

    score = _score_with_boxed_equivalence(completion_text, answer)
    if score > 0:
        return score

    try:
        return _score_with_math_verify_fallback(completion_text, answer)
    except Exception:
        return _score_with_simple_match(completion_text, answer)


def _parse_pass_ks(pass_k_arg: str | None, num_completions: int) -> list[int]:
    max_n = max(1, int(num_completions))
    if not pass_k_arg:
        return sorted({1, max_n})

    tokens = [x.strip().lower() for x in pass_k_arg.split(",") if x.strip()]
    if not tokens:
        return sorted({1, max_n})
    if "all" in tokens:
        return list(range(1, max_n + 1))

    ks = set()
    for token in tokens:
        try:
            k = int(token)
        except ValueError as e:
            raise ValueError(f"Invalid --pass-k token '{token}'") from e
        if k <= 0:
            raise ValueError("--pass-k values must be > 0")
        ks.add(min(k, max_n))
    return sorted(ks)


def _estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimate."""
    if n <= 0 or c <= 0:
        return 0.0
    k = min(max(1, k), n)
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def _score_results_with_pool(
    results: list[dict],
    answers: dict[int, Any],
    verifier_workers: int,
    verifier_timeout: float,
) -> tuple[dict[int, int], dict[str, int]]:
    score_by_result_idx: dict[int, int] = {}
    stats = {
        "tasks": 0,
        "timeouts": 0,
        "expired": 0,
        "exceptions": 0,
        "fallback_sync": 0,
    }

    tasks: list[tuple[int, tuple[str, Any]]] = []
    for ridx, row in enumerate(results):
        pidx = row.get("prompt_idx")
        if pidx in answers:
            tasks.append((ridx, (row.get("text", ""), answers[pidx])))
    stats["tasks"] = len(tasks)

    if not tasks:
        return score_by_result_idx, stats

    # If Pebble is unavailable, keep script usable with deterministic fallback.
    if ProcessPool is None:
        stats["fallback_sync"] = len(tasks)
        for ridx, payload in tasks:
            try:
                score_by_result_idx[ridx] = _score_with_archer(payload)
            except Exception:
                stats["exceptions"] += 1
                score_by_result_idx[ridx] = 0
        return score_by_result_idx, stats

    workers = max(1, int(verifier_workers))
    timeout_s = max(0.1, float(verifier_timeout))
    with ProcessPool(max_workers=workers, initializer=_pool_initializer) as pool:
        scheduled = [
            (ridx, pool.schedule(_score_with_archer, args=(payload,), timeout=timeout_s))
            for ridx, payload in tasks
        ]
        for ridx, future in scheduled:
            try:
                score_by_result_idx[ridx] = int(future.result())
            except FuturesTimeoutError:
                stats["timeouts"] += 1
                score_by_result_idx[ridx] = 0
            except ProcessExpired:
                stats["expired"] += 1
                score_by_result_idx[ridx] = 0
            except Exception:
                stats["exceptions"] += 1
                score_by_result_idx[ridx] = 0

    return score_by_result_idx, stats


def compute_scoring_metrics(
    results: list[dict],
    answers: dict[int, Any],
    pass_ks: list[int],
    verifier_workers: int,
    verifier_timeout: float,
    precomputed_scores: dict[int, int] | None = None,
    verifier_stats_override: dict[str, int] | None = None,
) -> dict:
    """
    Score all completions with the embedded verifier + Pebble pool, then compute:
    - mean@N
    - pass@k (unbiased estimator) for each requested k
    """
    by_prompt: dict[int, list[int]] = defaultdict(list)
    if precomputed_scores is None:
        score_by_result_idx, verifier_stats = _score_results_with_pool(
            results=results,
            answers=answers,
            verifier_workers=verifier_workers,
            verifier_timeout=verifier_timeout,
        )
    else:
        score_by_result_idx = precomputed_scores
        verifier_stats = verifier_stats_override or {
            "tasks": 0,
            "timeouts": 0,
            "expired": 0,
            "exceptions": 0,
            "fallback_sync": 0,
        }

    for ridx, row in enumerate(results):
        pidx = row.get("prompt_idx")
        if pidx not in answers:
            continue
        score = int(score_by_result_idx.get(ridx, 0))
        row["score"] = score
        by_prompt[pidx].append(score)

    if not by_prompt:
        return {
            "mean_at_n": 0.0,
            "pass_at_k": {str(k): 0.0 for k in pass_ks},
            "num_prompts_scored": 0,
            "per_prompt": {},
            "verifier_pool": verifier_stats,
        }

    per_prompt = {}
    for pidx, scores in sorted(by_prompt.items()):
        n = len(scores)
        correct = int(sum(scores))
        prompt_pass = {str(k): _estimate_pass_at_k(n, correct, k) for k in pass_ks}
        per_prompt[pidx] = {
            "n": n,
            "correct": correct,
            "mean": (correct / n) if n else 0.0,
            "pass_at_k": prompt_pass,
        }

    overall_mean = float(np.mean([v["mean"] for v in per_prompt.values()]))
    overall_pass = {
        str(k): float(np.mean([v["pass_at_k"][str(k)] for v in per_prompt.values()]))
        for k in pass_ks
    }
    return {
        "mean_at_n": overall_mean,
        "pass_at_k": overall_pass,
        "num_prompts_scored": len(per_prompt),
        "per_prompt": per_prompt,
        "verifier_pool": verifier_stats,
    }


@dataclass
class Stats:
    total_prompts: int = 0
    total_completions: int = 0
    finished_completions: int = 0
    total_tokens: int = 0
    nan_logprob_events: int = 0
    token_zero_events: int = 0
    nan_prompts: set = field(default_factory=set)
    nan_completions: set = field(default_factory=set)
    token_zero_prompts: set = field(default_factory=set)
    token_zero_completions: set = field(default_factory=set)
    errors: int = 0
    start_time: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def summary(self):
        elapsed = time.time() - self.start_time
        tps = self.total_tokens / elapsed if elapsed > 0 else 0
        print(f"\n{'='*72}")
        print(f"{BOLD}EVAL SUMMARY{RESET}")
        print(f"{'='*72}")
        print(f"  Prompts:          {self.total_prompts}")
        print(f"  Completions:      {self.finished_completions}/{self.total_completions}")
        print(f"  Total tokens:     {self.total_tokens}")
        print(f"  Throughput:       {tps:.1f} tok/s")
        print(f"  Elapsed:          {elapsed:.1f}s")
        print(f"  Errors:           {self.errors}")
        print()
        if self.nan_logprob_events > 0:
            print(f"  {RED}{BOLD}NaN logprob events:       {self.nan_logprob_events}{RESET}")
            print(f"  {RED}  Affected prompts:       {len(self.nan_prompts)}/{self.total_prompts}{RESET}")
            print(f"  {RED}  Affected completions:   {len(self.nan_completions)}/{self.total_completions}{RESET}")
        else:
            print(f"  {GREEN}NaN logprob events:       0 ✓{RESET}")
        print()
        if self.token_zero_events > 0:
            print(f"  {YELLOW}{BOLD}Token ID=0 events:        {self.token_zero_events}{RESET}")
            print(f"  {YELLOW}  Affected prompts:       {len(self.token_zero_prompts)}/{self.total_prompts}{RESET}")
            print(f"  {YELLOW}  Affected completions:   {len(self.token_zero_completions)}/{self.total_completions}{RESET}")
        else:
            print(f"  {GREEN}Token ID=0 events:        0 ✓{RESET}")
        print(f"{'='*72}")


async def stream_completion(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list,
    max_tokens: int,
    api_key: str,
    prompt_idx: int,
    completion_idx: int,
    stats: Stats,
    print_lock: asyncio.Lock,
) -> dict | None:
    """Stream a single completion, checking each token for anomalies."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "extra_body": {"top_k": -1},
        "stream": True,
        "logprobs": True,
        "top_logprobs": 1,
    }

    cid = f"P{prompt_idx:04d}/C{completion_idx}"
    tokens_text = []
    token_count = 0

    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=3600.0,
        ) as resp:
            if resp.status_code != 200:
                err = (await resp.aread()).decode()
                async with print_lock:
                    print(f"  {RED}[{cid}] HTTP {resp.status_code}: {err[:200]}{RESET}")
                async with stats._lock:
                    stats.errors += 1
                return None

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    tokens_text.append(content)

                logprobs_obj = choices[0].get("logprobs")
                if logprobs_obj is None:
                    continue

                lp_content = logprobs_obj.get("content")
                if not lp_content:
                    continue

                for tok_info in lp_content:
                    token_count += 1
                    async with stats._lock:
                        stats.total_tokens += 1

                    token_str = tok_info.get("token", "")
                    logprob = tok_info.get("logprob")
                    token_bytes = tok_info.get("bytes", [])

                    # ── NaN logprob detection ────────────────────────
                    is_nan = False
                    if logprob is None:
                        is_nan = True
                    elif isinstance(logprob, float) and (math.isnan(logprob) or math.isinf(logprob)):
                        is_nan = True

                    if is_nan:
                        async with stats._lock:
                            stats.nan_logprob_events += 1
                            stats.nan_prompts.add(prompt_idx)
                            stats.nan_completions.add((prompt_idx, completion_idx))
                        ctx = "".join(tokens_text[-5:])
                        async with print_lock:
                            print(
                                f"  {RED}{BOLD}[NaN LOGPROB]{RESET} "
                                f"{RED}{cid} tok#{token_count} "
                                f"token={repr(token_str)} logprob={logprob} "
                                f"bytes={token_bytes} "
                                f"ctx='...{ctx}'{RESET}"
                            )

                    # ── Token ID=0 detection ─────────────────────────
                    top_lps = tok_info.get("top_logprobs", [])
                    for entry in [tok_info] + top_lps:
                        tok_id = entry.get("token_id")
                        if tok_id is not None and tok_id == 0:
                            async with stats._lock:
                                stats.token_zero_events += 1
                                stats.token_zero_prompts.add(prompt_idx)
                                stats.token_zero_completions.add((prompt_idx, completion_idx))
                            async with print_lock:
                                print(
                                    f"  {YELLOW}{BOLD}[TOKEN_ID=0]{RESET} "
                                    f"{YELLOW}{cid} tok#{token_count} "
                                    f"token={repr(entry.get('token', ''))} "
                                    f"logprob={entry.get('logprob')} "
                                    f"ctx='...{''.join(tokens_text[-5:])}'{RESET}"
                                )

                    if token_bytes == [0] or (token_str == "" and token_bytes == []):
                        async with stats._lock:
                            stats.token_zero_events += 1
                            stats.token_zero_prompts.add(prompt_idx)
                            stats.token_zero_completions.add((prompt_idx, completion_idx))
                        async with print_lock:
                            print(
                                f"  {YELLOW}{BOLD}[TOKEN_ID=0?]{RESET} "
                                f"{YELLOW}{cid} tok#{token_count} "
                                f"token={repr(token_str)} bytes={token_bytes} "
                                f"logprob={logprob} "
                                f"ctx='...{''.join(tokens_text[-5:])}'{RESET}"
                            )

    except httpx.ReadTimeout:
        async with print_lock:
            print(f"  {RED}[{cid}] TIMEOUT{RESET}")
        async with stats._lock:
            stats.errors += 1
        return None
    except Exception as e:
        async with print_lock:
            print(f"  {RED}[{cid}] ERROR: {e}{RESET}")
        async with stats._lock:
            stats.errors += 1
        return None

    # ── Print completion result ──────────────────────────────────────
    async with stats._lock:
        stats.finished_completions += 1
        finished = stats.finished_completions
        total = stats.total_completions

    full_text = "".join(tokens_text)
    head = full_text[:400].replace("\n", "\\n")
    tail = full_text[-400:].replace("\n", "\\n")
    async with print_lock:
        print(f"  {GREEN}{BOLD}[DONE]{RESET} {GREEN}{cid} | {token_count} tok | [{finished}/{total}]{RESET}")
        print(f"    {GREEN}[HEAD] {head}{RESET}")
        if len(full_text) > 800:
            print(f"    {GREEN}[TAIL] ...{tail}{RESET}")

    return {
        "prompt_idx": prompt_idx,
        "completion_idx": completion_idx,
        "text": full_text,
        "token_count": token_count,
    }


async def progress_printer(stats: Stats, interval: float = 5.0):
    """Print aggregate progress on interval."""
    while True:
        await asyncio.sleep(interval)
        elapsed = time.time() - stats.start_time
        tps = stats.total_tokens / elapsed if elapsed > 0 else 0
        print(
            f"  {DIM}[PROGRESS] {stats.total_tokens} tok | {tps:.0f} tok/s | "
            f"{stats.finished_completions}/{stats.total_completions} done | "
            f"{stats.errors} err | {elapsed:.0f}s{RESET}",
            flush=True,
        )


async def run(args):
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print(f"{RED}No API key. Set OPENAI_API_KEY or pass --api-key{RESET}")
        sys.exit(1)
    try:
        pass_ks = _parse_pass_ks(args.pass_k, args.num_completions)
    except ValueError as e:
        print(f"{RED}Invalid --pass-k: {e}{RESET}")
        sys.exit(1)

    # ── Auto-detect model ────────────────────────────────────────────
    model = args.model
    if model is None:
        print(f"{CYAN}Auto-detecting model...{RESET}")
        async with httpx.AsyncClient() as c:
            r = await c.get(
                f"{args.base_url}/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        models = r.json()["data"]
        model = models[0]["id"]
        print(f"{CYAN}Using model: {model}{RESET}")

    # ── Load dataset ─────────────────────────────────────────────────
    if args.parquet:
        print(f"{CYAN}Loading parquet: {args.parquet}...{RESET}")
        df = load_dataset_parquet(args.parquet, args.num_prompts)
    else:
        hf_path = args.hf_dataset
        hf_split = args.hf_split
        print(f"{CYAN}Loading HuggingFace dataset: {hf_path} (split={hf_split})...{RESET}")
        df = load_dataset_hf(hf_path, split=hf_split, num_prompts=args.num_prompts)

    print(f"{CYAN}Loaded {len(df)} prompts × {args.num_completions} completions = {len(df) * args.num_completions} total{RESET}")

    # ── Build answers dict for scoring ───────────────────────────────
    answers = {}
    do_score = args.score
    if "answer" in df.columns and not args.no_score:
        do_score = True
        for idx, row in df.iterrows():
            answers[idx] = _to_native(row["answer"])
        print(f"{CYAN}Scoring enabled: {len(answers)} answers loaded (embedded verifier + Pebble pool){RESET}")
        ok, msg = _run_verifier_preflight()
        if ok:
            print(f"{CYAN}Verifier preflight: {msg}{RESET}")
        else:
            print(
                f"{YELLOW}Verifier preflight warning: {msg}. "
                "Will fallback to math_verify/simple matcher if needed."
                f"{RESET}"
            )
    elif args.score and "answer" not in df.columns:
        print(f"{RED}--score requires 'answer' column in dataset{RESET}")
        sys.exit(1)

    # ── Prepare all tasks ────────────────────────────────────────────
    stats = Stats()
    stats.start_time = time.time()
    stats.total_prompts = len(df)
    stats.total_completions = len(df) * args.num_completions

    print_lock = asyncio.Lock()

    # Pre-process all prompts
    prompts = []
    for idx, row in df.iterrows():
        # Support both pre-formatted message lists (prompt col) and raw problem text
        if "prompt" in df.columns:
            prompt = row["prompt"]
            if isinstance(prompt, str):
                try:
                    prompt = json.loads(prompt)
                except json.JSONDecodeError:
                    # Treat as raw text
                    prompt = [{"role": "user", "content": prompt + MATH_SUFFIX}]
            prompt = _to_native(prompt)
        elif "problem" in df.columns:
            prompt = [{"role": "user", "content": str(row["problem"]) + MATH_SUFFIX}]
        else:
            print(f"{RED}Dataset must have 'problem' or 'prompt' column{RESET}")
            sys.exit(1)

        extra = {}
        if "extra_info" in df.columns:
            extra = row.get("extra_info", "{}")
            if isinstance(extra, str):
                extra = json.loads(extra)
            extra = _to_native(extra)

        try:
            json.dumps(prompt)
        except TypeError as e:
            print(f"  {RED}SKIP prompt {idx}: not serializable: {e}{RESET}")
            continue

        prompts.append({
            "idx": idx,
            "messages": prompt,
            "data_source": row.get("data_source", "") if "data_source" in df.columns else "",
            "extra_info": extra,
        })

    # ── Launch all streams concurrently ──────────────────────────────
    sem = asyncio.Semaphore(args.concurrency)

    async def bounded_stream(client, p, n):
        async with sem:
            result = await stream_completion(
                client=client,
                base_url=args.base_url,
                model=model,
                messages=p["messages"],
                max_tokens=args.max_tokens,
                api_key=api_key,
                prompt_idx=p["idx"],
                completion_idx=n,
                stats=stats,
                print_lock=print_lock,
            )
            if result:
                result["data_source"] = p["data_source"]
                result["extra_info"] = p["extra_info"]
                result["prompt"] = p["messages"]
            return result

    verifier_pool_stats = {
        "tasks": 0,
        "timeouts": 0,
        "expired": 0,
        "exceptions": 0,
        "fallback_sync": 0,
    }
    score_by_result_idx: dict[int, int] = {}
    score_futures: dict[int, Any] = {}
    score_thread_tasks: dict[int, asyncio.Task] = {}
    score_pool = None
    score_pool_closed = False
    score_timeout_s = max(0.1, float(args.verifier_timeout))

    if do_score and answers and ProcessPool is not None:
        try:
            score_pool = ProcessPool(
                max_workers=max(1, int(args.verifier_workers)),
                initializer=_pool_initializer,
            )
        except Exception as e:
            print(
                f"{YELLOW}Warning: failed to create verifier pool ({e}). "
                f"Using async thread fallback for scoring.{RESET}"
            )
            score_pool = None
    elif do_score and answers:
        print(
            f"{YELLOW}Warning: pebble not available; using async thread fallback for scoring.{RESET}"
        )

    def _submit_score_job(result_idx: int, result_row: dict) -> None:
        if not (do_score and answers):
            return
        pidx = result_row.get("prompt_idx")
        if pidx not in answers:
            return

        verifier_pool_stats["tasks"] += 1
        payload = (result_row.get("text", ""), answers[pidx])

        if score_pool is None:
            verifier_pool_stats["fallback_sync"] += 1
            # Keep generation non-blocking even without Pebble.
            score_thread_tasks[result_idx] = asyncio.create_task(asyncio.to_thread(_score_with_archer, payload))
            return

        try:
            score_futures[result_idx] = score_pool.schedule(
                _score_with_archer,
                args=(payload,),
                timeout=score_timeout_s,
            )
        except Exception:
            verifier_pool_stats["exceptions"] += 1
            score_by_result_idx[result_idx] = 0

    async def _finalize_submitted_scores() -> None:
        nonlocal score_pool_closed
        try:
            for ridx, task in score_thread_tasks.items():
                try:
                    score_by_result_idx[ridx] = int(await task)
                except Exception:
                    verifier_pool_stats["exceptions"] += 1
                    score_by_result_idx[ridx] = 0

            for ridx, future in score_futures.items():
                try:
                    score_by_result_idx[ridx] = int(future.result())
                except FuturesTimeoutError:
                    verifier_pool_stats["timeouts"] += 1
                    score_by_result_idx[ridx] = 0
                except ProcessExpired:
                    verifier_pool_stats["expired"] += 1
                    score_by_result_idx[ridx] = 0
                except Exception:
                    verifier_pool_stats["exceptions"] += 1
                    score_by_result_idx[ridx] = 0
        finally:
            if score_pool is not None:
                try:
                    score_pool.close()
                    score_pool.join(timeout=5)
                except Exception:
                    pass
                score_pool_closed = True

    results = []
    async with httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=args.concurrency + 10,
            max_keepalive_connections=args.concurrency + 10,
        ),
    ) as client:
        progress_task = asyncio.create_task(progress_printer(stats, interval=args.progress_interval))

        tasks = []
        for p in prompts:
            for n in range(args.num_completions):
                tasks.append(asyncio.create_task(bounded_stream(client, p, n)))

        print(f"{CYAN}Launching {len(tasks)} streams (max concurrency {args.concurrency})...{RESET}\n")
        for task in asyncio.as_completed(tasks):
            try:
                r = await task
            except Exception as e:
                print(f"  {RED}Task exception: {e}{RESET}")
                stats.errors += 1
                continue

            if r is not None:
                results.append(r)
                _submit_score_job(len(results) - 1, r)

        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

    stats.summary()

    # ── Scoring ──────────────────────────────────────────────────────
    if do_score and answers:
        await _finalize_submitted_scores()
        score_info = compute_scoring_metrics(
            results=results,
            answers=answers,
            pass_ks=pass_ks,
            verifier_workers=args.verifier_workers,
            verifier_timeout=args.verifier_timeout,
            precomputed_scores=score_by_result_idx,
            verifier_stats_override=verifier_pool_stats,
        )
        mean_n = score_info["mean_at_n"]
        pass_k = score_info["pass_at_k"]
        n_scored = score_info["num_prompts_scored"]
        pool_info = score_info.get("verifier_pool", {})

        print(f"\n{'='*72}")
        print(f"{BOLD}SCORING (mean@{args.num_completions} + pass@k){RESET}")
        print(f"{'='*72}")
        print(f"  Prompts scored:   {n_scored}/{stats.total_prompts}")
        print(f"  {BOLD}mean@{args.num_completions}:          {mean_n:.4f} ({mean_n*100:.2f}%){RESET}")
        for k in pass_ks:
            p = float(pass_k.get(str(k), 0.0))
            print(f"  {BOLD}pass@{k}:                 {p:.4f} ({p*100:.2f}%){RESET}")
        print(
            "  Verifier pool:    "
            f"workers={args.verifier_workers}, timeout={args.verifier_timeout:.1f}s, "
            f"tasks={pool_info.get('tasks', 0)}, timeouts={pool_info.get('timeouts', 0)}, "
            f"expired={pool_info.get('expired', 0)}, exceptions={pool_info.get('exceptions', 0)}, "
            f"fallback_sync={pool_info.get('fallback_sync', 0)}"
        )
        print()

        # Per-prompt breakdown
        for pidx, info in score_info["per_prompt"].items():
            status = f"{GREEN}✓{RESET}" if info["mean"] > 0 else f"{RED}✗{RESET}"
            prompt_pass_n = info["pass_at_k"].get(str(args.num_completions), 0.0)
            print(
                f"    {status} P{pidx:04d}: {info['correct']}/{info['n']} correct "
                f"(mean={info['mean']:.2f}, pass@{args.num_completions}={prompt_pass_n:.2f})"
            )

        print(f"{'='*72}")

        # Attach score summary to output
        if args.output:
            output_data = {
                "score": score_info,
                "results": results,
            }
        else:
            output_data = None
    else:
        output_data = None

    # Safety cleanup for unexpected early exits before score finalization.
    if score_pool is not None and not score_pool_closed:
        try:
            score_pool.close()
            score_pool.join(timeout=5)
        except Exception:
            pass

    if args.output:
        payload = output_data if output_data else results
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\n{CYAN}Results saved to {args.output}{RESET}")


def main():
    parser = argparse.ArgumentParser(description="Async streaming eval with NaN/token0 debugging + math scoring")
    # Data source (mutually exclusive-ish: parquet takes priority)
    parser.add_argument("--parquet", type=str, default=None, help="Path to local eval parquet (overrides --hf-dataset)")
    parser.add_argument("--hf-dataset", type=str, default="MathArena/hmmt_nov_2025", help="HuggingFace dataset path")
    parser.add_argument("--hf-split", type=str, default="train", help="HuggingFace dataset split")

    # Endpoint
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="vLLM base URL")
    parser.add_argument("--model", type=str, default=None, help="Model name (auto-detected if not set)")
    parser.add_argument("--api-key", type=str, default=None, help="API key (or set OPENAI_API_KEY)")

    # Eval params
    parser.add_argument("-K", "--num-prompts", type=int, default=None, help="Number of prompts to eval (default: all)")
    parser.add_argument("-N", "--num-completions", type=int, default=4, help="Completions per prompt")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per completion")
    parser.add_argument("--concurrency", type=int, default=64, help="Max concurrent streams")
    parser.add_argument("--progress-interval", type=float, default=5.0, help="Progress print interval (seconds)")

    # Scoring
    parser.add_argument("--score", action="store_true", help="Force enable scoring (auto-enabled when 'answer' column exists)")
    parser.add_argument("--no-score", action="store_true", help="Disable scoring even if 'answer' column exists")
    parser.add_argument("--pass-k", type=str, default=None, help="Comma-separated pass@k values (e.g. '1,4,8' or 'all'). Default: 1 and N")
    parser.add_argument(
        "--verifier-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of Pebble verifier processes",
    )
    parser.add_argument(
        "--verifier-timeout",
        type=float,
        default=20.0,
        help="Per-completion verifier timeout in seconds",
    )

    # Output
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()