# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in Zaya prefix-cache correctness test."""

import os

import pytest

_SKIP_REASON = None
if os.getenv("RUN_SLOW") != "1":
    _SKIP_REASON = (
        "set RUN_SLOW=1 to run the Zaya prefix-cache integration test")
elif "VLLM_ZAYA_TEST_MODEL" not in os.environ:
    _SKIP_REASON = "set VLLM_ZAYA_TEST_MODEL to a local Zaya checkpoint"

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None,
                                reason=_SKIP_REASON or "")

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from tests.models.utils import check_logprobs_close  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.distributed import cleanup_dist_env_and_memory  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402

pytestmark = [
    pytestmark,
    pytest.mark.skipif(
        not current_platform.is_cuda(),
        reason="Zaya prefix-cache integration test requires CUDA"),
]

MODEL = os.getenv("VLLM_ZAYA_TEST_MODEL", "")
BLOCK_SIZE = int(os.getenv("VLLM_ZAYA_TEST_BLOCK_SIZE", "16"))
MAX_MODEL_LEN = int(os.getenv("VLLM_ZAYA_TEST_MAX_MODEL_LEN", "3072"))
EXPECTED_CACHED_TOKENS = int(
    os.getenv("VLLM_ZAYA_TEST_EXPECTED_CACHED_TOKENS", "2048"))
GPU_MEMORY_UTILIZATION = float(
    os.getenv("VLLM_ZAYA_TEST_GPU_MEMORY_UTILIZATION", "0.9"))
MOE_BACKEND = os.getenv("VLLM_ZAYA_TEST_MOE_BACKEND", "triton")

COMMON_PREFIX = (
    "You are an internal coding assistant helping an inference team debug "
    "a production prefix cache issue. The model is Zaya, the attention stack "
    "uses CCA for qkv projection, and every cached KV block must carry the "
    "matching CCA boundary state. Please reason carefully about state slots, "
    "block tables, and full-block cache hits. "
) * 36
WARM_PROMPT = (
    COMMON_PREFIX
    + "First request: summarize the cache warmup procedure in two short steps."
)
HIT_PROMPT = (
    COMMON_PREFIX
    + "Second request: explain how to verify that a prefix-cache hit is correct."
)
SAMPLING_PARAMS = SamplingParams(
    temperature=0.0,
    max_tokens=8,
    logprobs=5,
)


def _make_llm(*, enable_prefix_caching: bool) -> LLM:
    return LLM(
        model=MODEL,
        tokenizer=MODEL,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=True,
        enable_prefix_caching=enable_prefix_caching,
        moe_backend=MOE_BACKEND,
        seed=0,
    )


def _tuple(output):
    return (
        list(output.outputs[0].token_ids),
        output.outputs[0].text,
        output.outputs[0].logprobs,
    )


def test_zaya_prefix_cache_hit_matches_cold_cache():
    cold_llm = _make_llm(enable_prefix_caching=False)
    try:
        tokenizer = cold_llm.get_tokenizer()
        common_len = len(tokenizer.encode(COMMON_PREFIX))
        hit_len = len(tokenizer.encode(HIT_PROMPT))
        assert common_len >= EXPECTED_CACHED_TOKENS, common_len
        assert hit_len + SAMPLING_PARAMS.max_tokens <= MAX_MODEL_LEN, hit_len
        cold_output = cold_llm.generate([HIT_PROMPT], SAMPLING_PARAMS)[0]
    finally:
        del cold_llm
        cleanup_dist_env_and_memory()

    warm_llm = _make_llm(enable_prefix_caching=True)
    try:
        warm_llm.generate([WARM_PROMPT], SAMPLING_PARAMS)
        warm_output = warm_llm.generate([HIT_PROMPT], SAMPLING_PARAMS)[0]
    finally:
        del warm_llm
        cleanup_dist_env_and_memory()

    assert warm_output.num_cached_tokens >= EXPECTED_CACHED_TOKENS, (
        "expected a real prefix-cache hit, got "
        f"{warm_output.num_cached_tokens} cached tokens")
    check_logprobs_close(
        outputs_0_lst=[_tuple(cold_output)],
        outputs_1_lst=[_tuple(warm_output)],
        name_0="cold_cache",
        name_1="warm_prefix_cache",
        always_check_logprobs=True,
    )
