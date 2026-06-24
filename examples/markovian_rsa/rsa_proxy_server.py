# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI-compatible RSA chat completion proxy.

For serving locally:
```bash
    vllm serve Zyphra/ZAYA1-8B --port 8010 \
        --mamba-cache-dtype float32 --dtype bfloat16 \
        --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser zaya_xml

```
Then start this proxy in front of it:
```bash
    python examples/markovian_rsa/rsa_proxy_server.py \
        --upstream-base-url http://localhost:8010/v1 \
        --port 8001
```
Or, using Zyphracloud:
python examples/markovian_rsa/rsa_proxy_server.py \
    --upstream-base-url https://api.zyphracloud.com/api/v1 \
    --port 8001

Requests sent to http://localhost:8001/v1/chat/completions pass through
unchanged unless RSA parameters are present in the JSON body. RSA parameters are
stripped before forwarding to the upstream backend.
"""

import argparse
import asyncio
import copy
import json
import logging
import os
import random
import ssl
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("rsa_proxy_server")
logging.basicConfig(level=logging.INFO)

# Settings from Zaya1-8B Tech report: https://arxiv.org/abs/2605.05365
RSA_T_DEFAULT=8
RSA_K_DEFAULT=4
RSA_N_DEFAULT=32
SHORT_RESP_LEN_DEFAULT=32768
TAIL_WINDOW_LEN_DEFAULT=4096
MAX_TOKENS_DEFAULT=61440

HTTPX_TIMEOUT = httpx.Timeout(6 * 60 * 60)
UPSTREAM_TRANSPORT_RETRIES = 2
RSA_RESPONSE_HEARTBEAT_SECONDS = 10
SUMMARY_LEN = 4096
DEFAULT_SUMMARY_PROMPT = (
    "You are given a problem and a solution. Concisely summarize the solution "
    "into a chain-of-thought style outline that preserves all important "
    "information required to continue refinement later: the main approach(es), "
    "key steps, useful intermediate results, and any mistakes or dead ends. "
    f"Make your summary less than {SUMMARY_LEN} words while keeping the "
    "essential structure. If the candidate included a final answer, retain it "
    "at the end in \\boxed{}.\n"
)

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

RSA_PARAM_ALIASES = {
    "rsa_t": ("rsa_t",),
    "rsa_k": ("rsa_k",),
    "rsa_n": ("rsa_n",),
    "rsa_summarize": ("rsa_summarize", "summarize"),
    "rsa_tail_window": ("rsa_tail_window", "rsa_use_tail_window"),
    "tail_window_len": ("tail_window_len", "rsa_tail_window_len"),
    "short_resp_len": ("short_resp_len", "rsa_short_resp_len"),
    "use_pacore": ("use_pacore", "rsa_use_pacore"),
    "rsa_final_n": ("rsa_final_n", "final_aggregation_prompts"),
    "final_temperature": ("final_temperature", "rsa_final_temperature"),
    "final_top_p": ("final_top_p", "rsa_final_top_p"),
    "final_top_k": ("final_top_k", "rsa_final_top_k"),
    "summary_prompt": ("summary_prompt", "rsa_summary_prompt"),
    "return_aggregation_prompt": (
        "return_aggregation_prompt",
        "rsa_return_aggregation_prompt",
    ),
    "return_rsa_metadata": ("return_rsa_metadata", "rsa_return_metadata"),
    "rsa_max_concurrency": ("rsa_max_concurrency",),
    "rsa_debug": ("rsa_debug",),
}
RSA_PARAM_NAMES = {
    name for aliases in RSA_PARAM_ALIASES.values() for name in aliases
}


@dataclass
class RSAConfig:
    rsa_t: int
    rsa_k: int
    rsa_n: int
    rsa_summarize: bool
    rsa_tail_window: bool
    tail_window_len: int
    short_resp_len: int
    use_pacore: bool
    rsa_final_n: int
    final_temperature: float | None
    final_top_p: float | None
    final_top_k: int | None
    summary_prompt: str
    return_aggregation_prompt: bool
    return_rsa_metadata: bool
    rsa_max_concurrency: int
    rsa_debug: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("OpenAI-compatible RSA proxy server.")
    parser.add_argument(
        "--upstream-base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible upstream base URL, including /v1.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="Optional upstream API key. Request Authorization wins if present.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Proxy bind host.")
    parser.add_argument("--port", type=int, default=8001, help="Proxy bind port.")
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def _pop_alias(body: dict[str, Any], key: str, default: Any) -> Any:
    for alias in RSA_PARAM_ALIASES[key]:
        if alias in body:
            return body.pop(alias)
    return default


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def pop_rsa_config(body: dict[str, Any]) -> tuple[RSAConfig | None, dict[str, Any]]:
    upstream_body = copy.deepcopy(body)
    has_rsa_params = any(param in upstream_body for param in RSA_PARAM_NAMES)
    if not has_rsa_params:
        return None, upstream_body

    rsa_n_default = upstream_body.get("n", RSA_N_DEFAULT)
    config = RSAConfig(
        rsa_t=int(_pop_alias(upstream_body, "rsa_t", RSA_T_DEFAULT)),
        rsa_k=int(_pop_alias(upstream_body, "rsa_k", RSA_K_DEFAULT)),
        rsa_n=int(_pop_alias(upstream_body, "rsa_n", rsa_n_default)),
        rsa_summarize=parse_bool(_pop_alias(upstream_body, "rsa_summarize", False)),
        rsa_tail_window=parse_bool(_pop_alias(upstream_body, "rsa_tail_window", True)),
        tail_window_len=int(_pop_alias(upstream_body, "tail_window_len", TAIL_WINDOW_LEN_DEFAULT)),
        short_resp_len=int(_pop_alias(upstream_body, "short_resp_len", SHORT_RESP_LEN_DEFAULT)),
        use_pacore=parse_bool(_pop_alias(upstream_body, "use_pacore", False)),
        rsa_final_n=int(_pop_alias(upstream_body, "rsa_final_n", 1)),
        final_temperature=_optional_float(
            _pop_alias(upstream_body, "final_temperature", 0.6)
        ),
        final_top_p=_optional_float(_pop_alias(upstream_body, "final_top_p", 0.95)),
        final_top_k=_optional_int(_pop_alias(upstream_body, "final_top_k", 40)),
        summary_prompt=str(
            _pop_alias(upstream_body, "summary_prompt", DEFAULT_SUMMARY_PROMPT)
        ),
        return_aggregation_prompt=parse_bool(
            _pop_alias(upstream_body, "return_aggregation_prompt", False)
        ),
        return_rsa_metadata=parse_bool(
            _pop_alias(upstream_body, "return_rsa_metadata", True)
        ),
        rsa_max_concurrency=int(_pop_alias(upstream_body, "rsa_max_concurrency", 0)),
        rsa_debug=parse_bool(_pop_alias(upstream_body, "rsa_debug", False)),
    )
    return config, upstream_body


def count_tokens(text: str | None) -> int:
    return len((text or "").split())


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def _conversation_text(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = message.get("role", "user")
        content = _message_content_to_text(message.get("content"))
        if content:
            lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def extract_tail_segment(text: str | None, max_tokens: int) -> str:
    if not text or max_tokens <= 0:
        return text or ""
    return " ".join(text.split()[-max_tokens:]).strip()


def create_summarization_prompt(
    candidate_response: str,
    question: str,
    summarization_prompt: str,
) -> list[dict[str, str]]:
    if "</think>" in candidate_response:
        candidate_response = candidate_response.split("</think>")[0].strip()
    full_content = (
        f"{summarization_prompt}\n\n[PROBLEM]\n{question}\n\n"
        f"[CANDIDATE RESPONSE]\n{candidate_response}\n\n"
        "Now produce the concise, information-preserving summary."
    )
    return [{"role": "user", "content": full_content}]


def create_aggregation_prompt(
    original_chat_history: list[dict[str, Any]],
    candidate_summaries: list[str],
    use_pacore: bool = False,
) -> list[dict[str, str]]:
    question = _message_content_to_text(
        original_chat_history[0].get("content")
    ).split("Let's think")[0]
    parts = []
    if use_pacore:
        parts.append(
            "You are given a problem and a list of reference responses. "
            "Your job is to analyze these references and provide your own "
            "response. End with the final result in \\boxed{}.\n"
        )
    elif len(candidate_summaries) == 1:
        parts.append(
            "You are given a math problem and a candidate solution. "
            "The candidate may be incomplete or contain errors. Refine this "
            "trajectory and produce an improved, higher-quality solution. "
            "If it is entirely wrong, attempt a new strategy. End with the "
            "final result in \\boxed{}.\n"
        )
    else:
        parts.append(
            "You are given a math problem and several candidate solutions. "
            "Some candidates may be incorrect or contain errors. Aggregate the "
            "useful ideas and produce a single, high-quality solution. Reason "
            "carefully; if candidates disagree, choose the correct path. If all "
            "are incorrect, then attempt a different strategy. Do not restate "
            "the problem or the candidate solutions."
        )

    if use_pacore:
        parts.append("Original Problem:\n")
        parts.append(question.strip() + "\n")
        parts.append("Reference Responses:\n")
        for idx, answer in enumerate(candidate_summaries, 1):
            parts.append(f"Reference {idx} \n{(answer or '').strip()}\n")
        parts.append(
            "Now write a single improved solution. Provide clear reasoning and "
            "end with the final answer in \\boxed{}."
        )
    else:
        parts.append("[[ Problem ]]:\n")
        parts.append(question.strip() + "\n")
        if len(candidate_summaries) == 1:
            parts.append("[[ Candidate solutions (may contain mistakes) ]]\n")
            parts.append(
                f"---- Candidate ----\n{(candidate_summaries[0] or '').strip()}\n"
            )
            parts.append(
                "Now, based on the original problem and the reference response "
                "above, please provide your own comprehensive solution."
            )
        else:
            parts.append("[[ Candidate solutions (may contain mistakes) ]]:\n")
            for idx, answer in enumerate(candidate_summaries, 1):
                parts.append(f"---- Solution {idx} ----\n{(answer or '').strip()}\n")
            parts.append(
                "Now write a single improved solution. Provide clear reasoning "
                "and end with the final answer in \\boxed{}."
            )
    return [{"role": "user", "content": "\n".join(parts)}]


def extract_reasoning_from_response(
    response: str,
    use_pacore: bool,
    use_tail_window: bool,
    tail_window_len: int,
) -> str:
    reasoning = response or ""
    if use_pacore:
        if "</think>" in reasoning:
            reasoning = reasoning.split("</think>")[-1].strip()
        else:
            reasoning = reasoning.strip()
            tail_segment = extract_tail_segment(reasoning, tail_window_len)
            if tail_segment:
                reasoning = tail_segment
    else:
        if "</think>" in reasoning:
            reasoning = reasoning.split("</think>")[0].strip()
        else:
            reasoning = reasoning.strip()
        if use_tail_window and reasoning:
            tail_segment = extract_tail_segment(reasoning, tail_window_len)
            if tail_segment:
                reasoning = tail_segment
    return reasoning


def extract_summary_from_response(response: str) -> str:
    if "</think>" in response:
        return response.split("</think>")[-1].strip()
    return response.strip()


def _chat_choice_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise HTTPException(status_code=502, detail="Upstream returned no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content:
        return str(content)
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    return "" if reasoning is None else str(reasoning)


def _merge_chunk_delta(choice_state: dict[str, Any], delta: dict[str, Any]) -> None:
    message = choice_state.setdefault("message", {"role": "assistant", "content": ""})
    if delta.get("role"):
        message["role"] = delta["role"]
    if delta.get("content"):
        message["content"] = message.get("content", "") + str(delta["content"])
    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
    if reasoning:
        message["reasoning_content"] = (
            message.get("reasoning_content", "") + str(reasoning)
        )
    if delta.get("tool_calls"):
        message.setdefault("tool_calls", []).extend(delta["tool_calls"])


def _chat_completion_from_stream_chunks(
    chunks: list[dict[str, Any]],
    fallback_model: str | None,
) -> dict[str, Any]:
    choice_states: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] | None = None
    response_id: str | None = None
    created: int | None = None
    model: str | None = fallback_model

    for chunk in chunks:
        response_id = response_id or chunk.get("id")
        created = created or chunk.get("created")
        model = model or chunk.get("model")
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            index = int(choice.get("index", 0))
            choice_state = choice_states.setdefault(
                index,
                {
                    "index": index,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                },
            )
            delta = choice.get("delta") or {}
            if isinstance(delta, dict):
                _merge_chunk_delta(choice_state, delta)
            if choice.get("finish_reason") is not None:
                choice_state["finish_reason"] = choice["finish_reason"]
            if choice.get("logprobs") is not None:
                choice_state["logprobs"] = choice["logprobs"]

    if not choice_states:
        raise HTTPException(
            status_code=502,
            detail="Upstream stream returned no choices.",
        )

    return {
        "id": response_id or f"chatcmpl-stream-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created or int(time.time()),
        "model": model or "",
        "choices": [choice_states[index] for index in sorted(choice_states)],
        "usage": usage or {},
    }


def _completion_tokens(response: dict[str, Any], content: str) -> int:
    usage = response.get("usage") or {}
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    if output_tokens is not None:
        return int(output_tokens)
    return count_tokens(content)


def _debug_response_shape(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    return {
        "choice_count": len(choices),
        "message_keys": sorted(message),
        "content_len": len(str(message.get("content") or "")),
        "reasoning_content_len": len(str(message.get("reasoning_content") or "")),
        "reasoning_len": len(str(message.get("reasoning") or "")),
        "usage": response.get("usage") or {},
    }


def _add_usage(total: dict[str, int], response: dict[str, Any]) -> None:
    usage = response.get("usage") or {}
    total["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
    total["completion_tokens"] += int(
        usage.get("completion_tokens", usage.get("output_tokens")) or 0
    )
    total["total_tokens"] += int(usage.get("total_tokens") or 0)


def _normalize_usage(total: dict[str, int]) -> dict[str, int]:
    if total["total_tokens"] == 0:
        total["total_tokens"] = total["prompt_tokens"] + total["completion_tokens"]
    return total


def _request_with_messages(
    base_body: dict[str, Any],
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    request = copy.deepcopy(base_body)
    request["messages"] = copy.deepcopy(messages)
    request["n"] = 1
    request["stream"] = False
    if max_tokens is not None:
        if "max_completion_tokens" in request:
            request["max_completion_tokens"] = max_tokens
        else:
            request["max_tokens"] = max_tokens
    if temperature is not None:
        request["temperature"] = temperature
    if top_p is not None:
        request["top_p"] = top_p
    if top_k is not None:
        request["top_k"] = top_k
    return request


def _response_headers(headers: Any) -> dict[str, str]:
    return {
        key: value
        for key, value in dict(headers).items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def _error_chat_completion(exc: Exception, model: str | None) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-rsa-error-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or "",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"RSA proxy error: {exc!r}",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "rsa_error": True,
    }


async def _json_result_with_heartbeats(
    task: asyncio.Task[dict[str, Any]],
    model: str | None,
) -> AsyncIterator[bytes]:
    try:
        while not task.done():
            yield b"\n"
            await asyncio.sleep(RSA_RESPONSE_HEARTBEAT_SECONDS)
        try:
            result = await task
        except Exception as exc:
            logger.exception("Unhandled RSA proxy error after response started.")
            result = _error_chat_completion(exc, model)
        yield json.dumps(result).encode("utf-8")
    except asyncio.CancelledError:
        task.cancel()
        logger.warning("Client disconnected while RSA request was running.")
        raise


class RSAProxy:
    def __init__(self, upstream_base_url: str, api_key: str | None):
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.api_key = api_key

    def _upstream_url(self, path: str) -> str:
        clean_path = path.lstrip("/")
        if self.upstream_base_url.endswith("/v1") and clean_path.startswith("v1/"):
            clean_path = clean_path[len("v1/") :]
        return f"{self.upstream_base_url}/{clean_path}"

    def _headers(self, request: Request | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if request is not None:
            authorization = request.headers.get("authorization")
            if authorization:
                headers["Authorization"] = authorization
        if "Authorization" not in headers and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def post_json(
        self,
        path: str,
        body: dict[str, Any],
        request: Request | None,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            response = await client.post(
                self._upstream_url(path),
                json=body,
                headers=self._headers(request),
            )
            if not 200 <= response.status_code < 300:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=self._error_detail(response.text),
                )
            try:
                return response.json()
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=502,
                    detail="Upstream returned non-JSON response.",
                ) from exc

    async def post_chat_completion_json(
        self,
        body: dict[str, Any],
        request: Request | None,
        *,
        stream_upstream: bool = False,
    ) -> dict[str, Any]:
        if not stream_upstream:
            return await self.post_json("/chat/completions", body, request)

        stream_body = copy.deepcopy(body)
        stream_body["stream"] = True

        last_error: Exception | None = None
        for attempt in range(UPSTREAM_TRANSPORT_RETRIES + 1):
            try:
                return await self._post_chat_completion_stream_json(
                    stream_body,
                    request,
                    fallback_model=str(body.get("model", "")),
                )
            except (httpx.HTTPError, OSError, ssl.SSLError) as exc:
                last_error = exc
                if attempt >= UPSTREAM_TRANSPORT_RETRIES:
                    break
                logger.warning(
                    "Upstream stream transport error; retrying request "
                    "(attempt %s/%s): %s",
                    attempt + 1,
                    UPSTREAM_TRANSPORT_RETRIES,
                    repr(exc),
                )
                await asyncio.sleep(0.5 * (attempt + 1))

        raise HTTPException(
            status_code=502,
            detail=f"Upstream stream transport error: {last_error!r}",
        )

    async def _post_chat_completion_stream_json(
        self,
        stream_body: dict[str, Any],
        request: Request | None,
        *,
        fallback_model: str,
    ) -> dict[str, Any]:
        chunks: list[dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            async with client.stream(
                "POST",
                self._upstream_url("/chat/completions"),
                json=stream_body,
                headers=self._headers(request),
            ) as response:
                if not 200 <= response.status_code < 300:
                    content = (await response.aread()).decode(errors="replace")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=self._error_detail(content),
                    )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunks.append(json.loads(data))
                    except json.JSONDecodeError as exc:
                        raise HTTPException(
                            status_code=502,
                            detail=f"Upstream returned invalid SSE JSON: {data}",
                        ) from exc

        return _chat_completion_from_stream_chunks(
            chunks,
            fallback_model=fallback_model,
        )

    async def stream_json(
        self,
        path: str,
        body: dict[str, Any],
        request: Request,
    ) -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            async with client.stream(
                "POST",
                self._upstream_url(path),
                json=body,
                headers=self._headers(request),
            ) as response:
                if not 200 <= response.status_code < 300:
                    content = (await response.aread()).decode(errors="replace")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=self._error_detail(content),
                    )
                async for chunk in response.aiter_bytes():
                    yield chunk

    async def forward_raw(self, path: str, request: Request) -> Response:
        url = self._upstream_url(path)
        if request.url.query:
            url = f"{url}?{request.url.query}"

        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
        }
        if "authorization" not in {key.lower() for key in headers} and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = await request.body()
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            response = await client.request(
                request.method,
                url,
                content=data if data else None,
                headers=headers,
            )
            response_headers = _response_headers(response.headers)
            media_type = response.headers.get("content-type")
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
                media_type=media_type,
            )

    @staticmethod
    def _error_detail(content: str) -> Any:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content

    async def _generate_one_per_prompt(
        self,
        base_body: dict[str, Any],
        prompts: list[list[dict[str, Any]]],
        raw_request: Request,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_concurrency: int = 0,
    ) -> tuple[list[dict[str, Any]], list[str], list[int]]:
        semaphore = (
            asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None
        )

        async def infer_prompt(messages: list[dict[str, Any]]) -> dict[str, Any]:
            request_body = _request_with_messages(
                base_body,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            return await self.post_chat_completion_json(
                request_body,
                raw_request,
                stream_upstream=True,
            )

        async def infer_with_limit(messages: list[dict[str, Any]]) -> dict[str, Any]:
            if semaphore is None:
                return await infer_prompt(messages)
            async with semaphore:
                return await infer_prompt(messages)

        responses = await asyncio.gather(
            *(infer_with_limit(prompt) for prompt in prompts)
        )
        contents = [_chat_choice_content(response) for response in responses]
        token_counts = [
            _completion_tokens(response, content)
            for response, content in zip(responses, contents)
        ]
        return responses, contents, token_counts

    async def _summarize_candidates(
        self,
        base_body: dict[str, Any],
        original_messages: list[dict[str, Any]],
        candidate_responses: list[str],
        config: RSAConfig,
        raw_request: Request,
        total_usage: dict[str, int],
    ) -> tuple[list[str], list[int]]:
        question = _conversation_text(original_messages)
        prompts = [
            create_summarization_prompt(
                response,
                question,
                config.summary_prompt,
            )
            for response in candidate_responses
        ]
        response_payloads, responses, _ = await self._generate_one_per_prompt(
            base_body,
            prompts,
            raw_request,
            max_tokens=config.short_resp_len,
            max_concurrency=config.rsa_max_concurrency,
        )
        for response in response_payloads:
            _add_usage(total_usage, response)
        summaries = [extract_summary_from_response(response) for response in responses]
        summary_token_counts = [count_tokens(summary) for summary in summaries]
        return summaries, summary_token_counts

    async def run_rsa_chat_completion(
        self,
        base_body: dict[str, Any],
        config: RSAConfig,
        raw_request: Request,
    ) -> dict[str, Any]:
        original_messages = base_body.get("messages")
        if not isinstance(original_messages, list) or not original_messages:
            raise HTTPException(
                status_code=400,
                detail="RSA chat completions require a non-empty messages list.",
            )
        if config.rsa_n <= 0:
            raise HTTPException(status_code=400, detail="rsa_n must be >= 1.")
        if config.rsa_k <= 0:
            raise HTTPException(status_code=400, detail="rsa_k must be >= 1.")
        if config.rsa_max_concurrency < 0:
            raise HTTPException(
                status_code=400,
                detail="rsa_max_concurrency must be >= 0.",
            )
        if config.rsa_tail_window and config.rsa_summarize:
            raise HTTPException(
                status_code=400,
                detail="rsa_tail_window requires rsa_summarize=false.",
            )
        if config.rsa_t <= 0:
            return await self.post_json("/chat/completions", base_body, raw_request)

        final_prompts_count = config.rsa_final_n
        if final_prompts_count <= 0:
            final_prompts_count = config.rsa_n
        final_prompts_count = min(final_prompts_count, config.rsa_n)

        population: list[str] | None = None
        population_token_counts: list[int] | None = None
        population_token_counts_before_final: list[int] | None = None
        iteration_responses_history: list[list[str]] = []
        iteration_token_counts_history: list[list[int]] = []
        iteration_prompt_counts: list[int] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        final_response_payloads: list[dict[str, Any]] = []
        final_aggregation_prompts: list[list[dict[str, Any]]] = []

        logger.info(
            "Running RSA: T=%s, G=%s, K=%s, summarize=%s, final_n=%s",
            config.rsa_t,
            config.rsa_n,
            config.rsa_k,
            config.rsa_summarize,
            final_prompts_count,
        )

        for iteration in range(config.rsa_t + 1):
            is_init_step = iteration == 0
            is_final_iteration = iteration == config.rsa_t

            if population is not None:
                if config.rsa_summarize:
                    population, population_token_counts = await self._summarize_candidates(
                        base_body,
                        original_messages,
                        population,
                        config,
                        raw_request,
                        total_usage,
                    )
                else:
                    population = [
                        extract_reasoning_from_response(
                            response,
                            config.use_pacore,
                            config.rsa_tail_window,
                            config.tail_window_len,
                        )
                        for response in population
                    ]
                    population_token_counts = [
                        count_tokens(response) for response in population
                    ]

            if is_final_iteration and population_token_counts is not None:
                population_token_counts_before_final = list(population_token_counts)

            prompts_per_sample = (
                final_prompts_count
                if is_final_iteration and not is_init_step
                else config.rsa_n
            )
            if is_init_step:
                prompts = [copy.deepcopy(original_messages) for _ in range(config.rsa_n)]
            else:
                if population is None:
                    raise HTTPException(
                        status_code=500,
                        detail="Internal RSA state error: missing population.",
                    )
                prompts = []
                for prompt_idx in range(prompts_per_sample):
                    if config.rsa_k == 1:
                        subsampled = [population[prompt_idx % len(population)]]
                    else:
                        sample_size = min(config.rsa_k, len(population))
                        k_indices = random.sample(range(len(population)), sample_size)
                        subsampled = [population[idx] for idx in k_indices]
                    prompts.append(
                        create_aggregation_prompt(
                            original_messages,
                            subsampled,
                            use_pacore=config.use_pacore,
                        )
                    )

            iteration_prompt_counts.append(prompts_per_sample)
            max_tokens = (
                config.short_resp_len
                if not config.rsa_summarize and not is_final_iteration
                else None
            )
            temperature = (
                config.final_temperature
                if is_final_iteration and not is_init_step
                else None
            )
            top_p = (
                config.final_top_p
                if is_final_iteration and not is_init_step
                else None
            )
            top_k = (
                config.final_top_k
                if is_final_iteration and not is_init_step
                else None
            )
            response_payloads, responses, token_counts = await self._generate_one_per_prompt(
                base_body,
                prompts,
                raw_request,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_concurrency=config.rsa_max_concurrency,
            )
            for response in response_payloads:
                _add_usage(total_usage, response)
            if config.rsa_debug and response_payloads:
                logger.info(
                    "RSA iteration %s first response shape: %s",
                    iteration,
                    _debug_response_shape(response_payloads[0]),
                )

            iteration_responses_history.append(list(responses))
            iteration_token_counts_history.append(list(token_counts))
            population = responses
            population_token_counts = token_counts
            final_response_payloads = response_payloads
            if is_final_iteration and not is_init_step:
                final_aggregation_prompts = prompts

            logger.info(
                "RSA iteration %s/%s completed: %s responses, avg output tokens=%.1f",
                iteration,
                config.rsa_t,
                len(responses),
                _mean([float(token_count) for token_count in token_counts]),
            )

        return self._build_rsa_response(
            final_response_payloads,
            config,
            iteration_responses_history,
            iteration_token_counts_history,
            iteration_prompt_counts,
            population_token_counts_before_final,
            final_aggregation_prompts,
            _normalize_usage(total_usage),
        )

    def _build_rsa_response(
        self,
        final_response_payloads: list[dict[str, Any]],
        config: RSAConfig,
        iteration_responses_history: list[list[str]],
        iteration_token_counts_history: list[list[int]],
        iteration_prompt_counts: list[int],
        population_token_counts_before_final: list[int] | None,
        final_aggregation_prompts: list[list[dict[str, Any]]],
        total_usage: dict[str, int],
    ) -> dict[str, Any]:
        if not final_response_payloads:
            raise HTTPException(status_code=502, detail="RSA produced no final response.")

        response = copy.deepcopy(final_response_payloads[0])
        choices = []
        for payload in final_response_payloads:
            for choice in payload.get("choices", []):
                next_choice = copy.deepcopy(choice)
                next_choice["index"] = len(choices)
                choices.append(next_choice)
        response["choices"] = choices
        response["usage"] = total_usage
        response["id"] = f"chatcmpl-rsa-{int(time.time() * 1000)}"

        token_count_history = [
            _mean([float(count) for count in iter_counts])
            for iter_counts in iteration_token_counts_history
        ]
        metadata: dict[str, Any] = {
            "rsa_iterations": config.rsa_t,
            "rsa_k": config.rsa_k,
            "rsa_n": config.rsa_n,
            "rsa_summarize": config.rsa_summarize,
            "rsa_tail_window": config.rsa_tail_window,
            "tail_window_len": config.tail_window_len,
            "short_resp_len": config.short_resp_len,
            "use_pacore": config.use_pacore,
            "rsa_max_concurrency": config.rsa_max_concurrency,
            "rsa_debug": config.rsa_debug,
            "rsa_token_count_history": token_count_history,
        }
        if config.return_rsa_metadata:
            response["rsa_metadata"] = metadata
        if config.return_aggregation_prompt:
            aggregation_prompt = (
                final_aggregation_prompts[0] if final_aggregation_prompts else None
            )
            response["aggregation_prompt"] = aggregation_prompt
            if config.return_rsa_metadata:
                metadata["rsa_final_aggregation_prompt"] = aggregation_prompt
        return response


def build_app(proxy: RSAProxy) -> FastAPI:
    app = FastAPI()

    @app.middleware("http")
    async def close_client_connections(request: Request, call_next: Any) -> Response:
        response = await call_next(request)
        response.headers["Connection"] = "close"
        return response

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def create_chat_completion(raw_request: Request) -> Response:
        try:
            body = await raw_request.json()
            config, upstream_body = pop_rsa_config(body)
            if config is None:
                if body.get("stream"):
                    return StreamingResponse(
                        proxy.stream_json("/chat/completions", body, raw_request),
                        media_type="text/event-stream",
                    )
                return JSONResponse(
                    await proxy.post_json("/chat/completions", body, raw_request)
                )

            if body.get("stream"):
                raise HTTPException(
                    status_code=400,
                    detail="RSA chat completions do not support stream=true.",
                )

            task = asyncio.create_task(
                proxy.run_rsa_chat_completion(
                    upstream_body,
                    config,
                    raw_request,
                )
            )
            return StreamingResponse(
                _json_result_with_heartbeats(
                    task,
                    str(upstream_body.get("model", "")),
                ),
                media_type="application/json",
            )
        except HTTPException:
            raise
        except asyncio.CancelledError:
            logger.warning("Client disconnected while RSA request was running.")
            raise
        except Exception as exc:
            logger.exception("Unhandled RSA proxy error.")
            return JSONResponse(
                {"detail": f"Unhandled RSA proxy error: {exc!r}"},
                status_code=500,
            )

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def passthrough(path: str, raw_request: Request) -> Response:
        return await proxy.forward_raw(path, raw_request)

    return app


def main() -> None:
    args = parse_args()
    proxy = RSAProxy(args.upstream_base_url, args.api_key)
    app = build_app(proxy)
    uvicorn.run(app, host=args.host, port=args.port, loop="uvloop")


if __name__ == "__main__":
    main()
