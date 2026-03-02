from openai import AsyncOpenAI
from copy import deepcopy
import argparse
import polars as pl
import os
import sys
from typing import Optional
import re
import asyncio
import logging
from typing import Any, Tuple, AsyncIterable, Callable

# from transformers import AutoTokenizer
from dataclasses import dataclass
import glob
import json
import httpx
import random


# ---- Schema definition ----
# Schema for generated columns only (answer, usage)
# Other columns (messages, ground_truth, data_source, env_class) use types from input dataset
GENERATED_COLUMNS_SCHEMA = {
    "answer": pl.Utf8,  # String answer (generated)
    "usage": pl.Struct([
        pl.Field("reasoning_tokens", pl.Int64),
        pl.Field("content_tokens", pl.Int64),
        pl.Field("output_tokens", pl.Int64),
        pl.Field("input_tokens", pl.Int64),
        pl.Field("total_tokens", pl.Int64),
    ]),}

SERIALIZED_COLS = {"ground_truth", "messages"}  # enforce JSON strings for these columns


# ---- Helper functions ----



def _json_dumps_or_none(x: Any) -> Optional[str]:
    if x is None:
        return None
    # compact + stable, keeps unicode readable
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), sort_keys=True)



# helper class for retry, set params and then call with_retries(coro_fn)
class Retry:
    def __init__(self, max_attempts: int=5, base_delay: float=0.5, max_delay: float=20.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retry_status = {429, 500, 502, 503, 504}
        
    async def with_retries(self, coro_fn):
        attempt = 0
        while True:
            attempt += 1
            try:
                return await coro_fn()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status not in self.retry_status or attempt >= self.max_attempts:
                    raise
            except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError, httpx.PoolTimeout):
                if attempt >= self.max_attempts:
                    raise
            # exponential backoff with jitter
            delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
            delay *= (0.5 + random.random())  # jitter in [0.5, 1.5)
            delay = min(delay, self.max_delay)  # cap at max_delay after jitter
            await asyncio.sleep(delay)
            
        return None # should never reach here


def render_readable_dict(d: dict, indent: int = 2) -> str:
    return _render(d, indent=indent, level=0)


def _render(x: Any, indent: int, level: int) -> str:
    pad = " " * (indent * level)
    pad_in = " " * (indent * (level + 1))

    if x is None: return "null"
    if x is True: return "true"
    if x is False: return "false"
    if isinstance(x, (int, float)) and not isinstance(x, bool): return str(x)

    if isinstance(x, str):
        return _render_string_keep_quotes(x, indent=indent, level=level)

    if isinstance(x, list):
        if not x: return "[]"
        items = [f"{pad_in}{_render(v, indent, level + 1)}" for v in x]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"

    if isinstance(x, dict):
        if not x: return "{}"
        parts = []
        for k, v in x.items():
            key = json.dumps(k, ensure_ascii=False)  # quoted keys like JSON
            val = _render(v, indent, level + 1)

            # If value renders multi-line, put it after ": " (still aligned nicely)
            if "\n" in val:
                parts.append(f"{pad_in}{key}: {val}")
            else:
                parts.append(f"{pad_in}{key}: {val}")

        return "{\n" + ",\n".join(parts) + "\n" + pad + "}"

    # fallback for non-JSON types: represent them as strings, quoted
    return _render_string_keep_quotes(str(x), indent=indent, level=level)


def _render_string_keep_quotes(s: str, indent: int, level: int) -> str:
    # Normalize newline display
    s_norm = s.replace("\r\n", "\n").replace("\r", "\n")

    # Escape backslashes and quotes so the output is readable/unambiguous
    s_vis = s_norm.replace("\\", "\\\\").replace('"', '\\"')

    # Single-line: normal quoted string
    if "\n" not in s_vis:
        return f"\"{s_vis}\""

    # Multi-line: "hanging quote" style, indented continuation lines
    cont_pad = " " * (indent * (level + 1))
    lines = s_vis.split("\n")

    out = [f"\"{lines[0]}"]
    out += [f"{cont_pad}{line}" for line in lines[1:]]
    out[-1] = out[-1] + "\""
    return "\n".join(out)

# ---- Serialization config (add near your globals) ----


def _json_dumps_or_none(x: Any) -> Optional[str]:
    if x is None:
        return None
    # compact + stable, keeps unicode readable
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def concatenate_parquet_chunks(output_dir: str) -> list[str]:
    output_files = sorted(glob.glob(os.path.join(output_dir, "chunk_*.parquet")))
    if not output_files:
        print(f"No chunk parquet files found in {output_dir}")
        return []

    dfs = []
    for file in output_files:
        df = pl.read_parquet(file)
        print(
            f"Reading {os.path.basename(file)}: {len(df)} rows, "
            f"{len(df.columns)} columns: {df.columns}"
        )

        # hard-enforce: serialized cols are Utf8, generated cols are typed
        exprs = []
        for c in SERIALIZED_COLS:
            if c in df.columns:
                exprs.append(pl.col(c).cast(pl.Utf8))

        # keep these consistent too
        if "answer" in df.columns:
            exprs.append(pl.col("answer").cast(pl.Utf8))
        if "usage" in df.columns:
            exprs.append(pl.col("usage").cast(GENERATED_COLUMNS_SCHEMA["usage"]))

        if exprs:
            df = df.with_columns(exprs)

        dfs.append(df)

    out = pl.concat(dfs, how="diagonal_relaxed")
    output_path = os.path.join(output_dir, "all_samples.parquet")
    out.write_parquet(output_path)

    print(f"Concatenated {len(out)} samples into {output_path}")
    return output_files


class Sample:
    """
    Sample class for a single sample from the dataset.
    
    Intended to be constructed using a named row dict from the dataset. Dataset must conform to either of these formats:
    
    SkyRL convention OR verl convention.
    
    everything optional is consolidated into metadata
    everything from verl that conflicts with SkyRL is consolidated into skyRL convention
    "prompt" becomes messages
    
    if reward_model.style (verl) or reward_spec.method (SkyRL) is not "rule" then we raise NotImplementedError (for now)
    """
    def __init__(self, row: dict):
        """
        Args:
            row: Named row dict from the dataset
        """

        self.messages = None
        
        # directly from API response, will be parsed as needed for scoring and writing to disk
        self.extra_info = dict()
        self.ground_truth = None
        self.reward_method = None
        self.data_source = None
        self.env_class = ""
        
        self.from_row(row)
        
    def from_row(self, row: dict):
        """
        Detect the convention of the row and transform it to the format expected by the EvalManager/InferenceManager.
        """
        if row.get("reward_model", None) is not None:
            self.transform_verl(row)
        elif row.get("reward_spec", None) is not None:
            self.transform_skyrl(row)
        else:
            raise ValueError(f"Unknown convention: {row}")
        
    def transform_verl(self, row: dict):
        """
        Transform a row dict from the verl convention to the format expected by the EvalManager/InferenceManager.
        
        verl convention:
        data = {
            "data_source": data_source,     # String: Name/identifier of the data source. REQUIRED
            "prompt": [{                    # List: Conversation format. REQUIRED
                "role": "user",
                "content": question
            }],
            "ability": "math",              # String: Ability identifier. OPTIONAL
            "reward_model": {
                "style": "rule",            # String: Either "rule" or "model". REQUIRED
                "ground_truth": solution    # Expected solution. REQUIRED
            },
            "extra_info": {
                'split': split,             # String: Split identifier. OPTIONAL
                'index': idx                # Integer: Index identifier. OPTIONAL
            }
        }
        """
        self.messages = row.get("prompt", None)
        reward_model = row.get("reward_model", None)
        self.ground_truth = reward_model.get("ground_truth", None) if reward_model else None
        self.extra_info = row.get("extra_info", None)
        self.data_source = row.get("data_source", None)
    
    def transform_skyrl(self, row: dict):
        """
        Transform a row dict from the SkyRL convention to the format expected by the EvalManager/InferenceManager.
        
        SkyRL convention:
        data = {
            "data_source": data_source,     # String: Name/identifier of the data source. REQUIRED
            "prompt": [                     # List: Conversation format. REQUIRED
                {
                    "role": "user",            
                    "content": question,       
                }
            ],
            "env_class": env_class,         # String: Environment class identifier. REQUIRED
            "reward_spec": {
                "method": "rule",           # String: Either "rule" or "reward_model". REQUIRED
                "ground_truth": solution,   # Expected solution. REQUIRED
            },
            "extra_info": {                 # Dict: Additional metadata. OPTIONAL
                # ... add your own fields here
            },
        }
        """
        self.messages = row.get("prompt", None)
        reward_spec = row.get("reward_spec", None)
        self.ground_truth = reward_spec.get("ground_truth", None) if reward_spec else None
        self.reward_method = reward_spec.get("method", None) if reward_spec else None
        self.extra_info = row.get("extra_info", None)
        self.data_source = row.get("data_source", None)
        self.env_class = row.get("env_class", None)

    def get_answer(self) -> str:
        """
        Extract the answer from the last ASSISTANT message.
        Returns the content field from the last message in self.messages, or empty string if not found.
        """
        if self.messages is None:
            return ""
        for msg in reversed(self.messages):
            if msg.get("role", None) == "assistant":
                return msg.get("content", "")
            
        return ""

    def get_total_response_length(self) -> int:
        """
        sum all content and reasoning lengths in self.messages
        """
        return sum(len(msg.get("content", "")) + len(msg.get("reasoning", "")) for msg in self.messages)

    def to_row_dict(self) -> dict:
        """
        Should have messages, ground_truth fields/cols. Consumed in a generator for polars to write to parquet from the completed queue
        messages should be a list of dicts with role and content keys, and the response should contain reasoning and content keys (either may be empty)
        """
        # Extract usage from metadata if present
        usage = None
        if self.extra_info and "usage" in self.extra_info:
            usage = self.extra_info["usage"]
                
        return {
            "messages": self.messages,  
            "ground_truth": self.ground_truth,
            "data_source": self.data_source,
            "env_class": self.env_class,
            "answer": self.get_answer(),
            "usage": usage,
        }        
        
    def to_readable_string(self) -> str:
        """
        Render the sample as a readable string.
        """
        return render_readable_dict(self.to_row_dict())


class InferenceManager:
    """
    Async compatible class for abstracting minor differences in API response formats for inference. Uses semaphore for concurrency control in requests, 
    and a process pool to parse responses when needed (as with vllm think token splitting)
    
    if using vllm, reasoning content is not separate in responses of API, so the inference manager will attempt to infer think tokens from local 
    instance of tokenizer (in string-space, not token ID space, but still verify is a single token)
    
    Example OpenAI chat completion response object:
        {
            "id": "resp_67cb71b351908190a308f3859487620d06981a8637e6bc44",
            "object": "response",
            "created_at": 1741386163,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-4o-2024-08-06",
            "output": [
                {
                "type": "message",
                "id": "msg_67cb71b3c2b0819084d481baaaf148f206981a8637e6bc44",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                    "type": "output_text",
                    "text": "Silent circuits hum,  \nThoughts emerge in data streams—  \nDigital dawn breaks.",
                    "annotations": []
                    }
                ]
                }
            ],
            "parallel_tool_calls": true,
            "previous_response_id": null,
            "reasoning": {
                "effort": null,
                "summary": null
            },
            "store": true,
            "temperature": 1.0,
            "text": {
                "format": {
                "type": "text"
                }
            },
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
            "truncation": "disabled",
            "usage": {
                "input_tokens": 32,
                "input_tokens_details": {
                "cached_tokens": 0
                },
                "output_tokens": 18,
                "output_tokens_details": {
                "reasoning_tokens": 0
                },
                "total_tokens": 50
            },
            "user": null,
            "metadata": {}
            }

    Example openrouter response object:
    
    {
        "id": "resp_1234567890",
        "object": "response",
        "created_at": 1234567890,
        "model": "openai/o4-mini",
        "output": [
            {
            "type": "reasoning",
            "id": "rs_abc123",
            "encrypted_content": "gAAAAABotI9-FK1PbhZhaZk4yMrZw3XDI1AWFaKb9T0NQq7LndK6zaRB...",
            "summary": [
                "First, I need to determine the current year",
                "Then calculate the difference from 1995",
                "Finally, compare that to 30 years"
            ]
            },
            {
            "type": "message",
            "id": "msg_xyz789",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                "type": "output_text",
                "text": "Yes. In 2025, 1995 was 30 years ago. In fact, as of today (Aug 31, 2025), it's exactly 30 years since Aug 31, 1995.",
                "annotations": []
                }
            ]
            }
        ],
        "usage": {
            "input_tokens": 15,
            "output_tokens": 85,
            "output_tokens_details": {
            "reasoning_tokens": 45
            },
            "total_tokens": 100
        },
        "status": "completed"
        }
        
    """
    def __init__(self, http_endpoint: str, model_name: str, api_key_env: str, max_requests: int=16, max_attempts: int=5, base_delay: float=0.5, max_delay: float=20.0, temperature: float=1.0, max_tokens: int=1024, top_p: float=1.0, top_k: int=0, n: int=1):
        self.base_url = self.get_base_url(http_endpoint)
        self.model_name = model_name
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.n = n
        self.semaphore = asyncio.Semaphore(max_requests)
        
        # set timeout for requests, if not set then defaults to low value which is not ideal for long generations
        timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
        
        # create client
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=os.getenv(self.api_key_env), timeout=timeout)
        
        self.special_token_ids = set()
        self.special_token_texts = set()
        self.tokenizer = None
        
        self.reasoning_warning_issued = False
        
        # Initialize Retry instance for inference operations
        self.retry = Retry(max_attempts=max_attempts, base_delay=base_delay, max_delay=max_delay)
    
    @staticmethod
    def get_base_url(http_endpoint: str) -> str:
        """
        Get the base URL for the API.
        """
        # The OpenAI client expects base_url to end with /v1 (it appends /chat/completions)
        # If endpoint is http://vllm:8000/v1/chat/completions, we want http://vllm:8000/v1
        base_url = http_endpoint.rstrip('/')
        if base_url.endswith('/v1/chat/completions'):
            base_url = base_url[:-len('/chat/completions')]
        elif base_url.endswith('/chat/completions'):
            # If it ends with /chat/completions but not /v1/chat/completions, add /v1
            base_url = base_url[:-len('/chat/completions')] + '/v1'
        elif not base_url.endswith('/v1'):
            # If it's just the base URL, add /v1
            base_url = base_url.rstrip('/') + '/v1'
            
        print(f"Using base URL: {base_url}")
        return base_url

    async def infer(self, sample: Sample) -> list[Sample]:
        """
        Infer a sample using the API.
        """
        async def _create_completion():
            return await self.client.chat.completions.create(
                model=self.model_name,
                messages=sample.messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                n=self.n,
                extra_body={"top_k": self.top_k} if self.top_k > 0 else None,
            )

        async with self.semaphore:
            response = await self.retry.with_retries(_create_completion)
            
            # Extract token usage information from response
            usage_info = self.extract_token_usage(response)
            
            # generator may return multiple samples per prompt, so we need to return a list of samples
            samples = []
            
            # iterate over response choices and for each "assistant" message split if desired, assign/update sample messages dict directly
            for item in response.choices:
                new_sample = deepcopy(sample)

                if item.message.role == "assistant":
                    new_sample = self.parse_assistant_message(item.message, new_sample)
                else:
                    # pass through non-assistant messages as is - convert to dict format
                    new_sample.messages.append({
                        "role": item.message.role,
                        "content": getattr(item.message, "content", ""),
                    })
                
                # Add usage information to metadata
                if new_sample.extra_info is None:
                    new_sample.extra_info = {}
                new_sample.extra_info["usage"] = usage_info
                
                samples.append(new_sample)
                
            return samples

    def initialize_tokenizer(self):
        """
        Initialize the tokenizer for the model and infer the think token IDs and texts.
        """
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.special_token_ids = set(self.tokenizer.all_special_ids)
        self.special_token_texts = set(self.tokenizer.all_special_tokens)
        
    def extract_token_usage(self, response: Any) -> dict:
        """
        Extract token usage information from the API response object.
        Returns a dict with reasoning_tokens and content_tokens (token ID lengths, not using tokenizer).
        
        Attempts to extract from response.usage.output_tokens_details.reasoning_tokens if available,
        otherwise infers from response.usage.output_tokens.
        """
        usage_dict = {
            "reasoning_tokens": None,
            "content_tokens": None,
            "output_tokens": None,
            "input_tokens": None,
            "total_tokens": None,
        }
        
        try:
            usage = response.usage
            if usage is None:
                return usage_dict
            
            # Extract basic usage info
            usage_dict["output_tokens"] = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
            usage_dict["input_tokens"] = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
            usage_dict["total_tokens"] = getattr(usage, "total_tokens", None)
            
            # Try to extract reasoning tokens from output_tokens_details
            reasoning_tokens = None
            output_tokens_details = getattr(usage, "output_tokens_details", None)
            if output_tokens_details is not None:
                reasoning_tokens = getattr(output_tokens_details, "reasoning_tokens", None)
            
            # If reasoning_tokens is available, calculate content_tokens
            if reasoning_tokens is not None and usage_dict["output_tokens"] is not None:
                usage_dict["reasoning_tokens"] = reasoning_tokens
                usage_dict["content_tokens"] = usage_dict["output_tokens"] - reasoning_tokens
            elif usage_dict["output_tokens"] is not None:
                # If reasoning_tokens is not available, we can't distinguish
                # Set both to None to indicate we don't know the breakdown
                usage_dict["reasoning_tokens"] = None
                usage_dict["content_tokens"] = None
                
        except Exception as e:
            logging.warning(f"Failed to extract token usage from response: {e}")
            
        return usage_dict
        
    def parse_assistant_message(self, message: Any, sample: Sample) -> Sample:
        """
        Parse an assistant message into content and reasoning by splitting on the end-think token if specified.
        
        We want to handle all cases of think splitting, including:
        - reasoning already present, do not split
        - reasoning not present, split on think open token (in prompt/user message from Sample) and think close token (in response message)
            - think open token not found, assume all response will be non-reasoning content
            - think close token not found, assume all response will be reasoning content
            
        openrouter uses this syntax:
        console.log('REASONING:', response.choices[i].message.reasoning);
        console.log('CONTENT:', response.choices[i].message.content);
        """
        # new message content to append in Sample
        sample.messages.append({
            "role": "assistant",
            "content": "",
            "reasoning": "",
        })

        # Extract content + reasoning from message object (attrs, not dict keys)
        message_content = getattr(message, "content", None) or ""

        # OpenRouter uses `reasoning`; vLLM often uses `reasoning_content`
        message_reasoning = getattr(message, "reasoning", None)
        if message_reasoning is None:
            message_reasoning = getattr(message, "reasoning_content", None)

        # first check if reasoning content is already present in response message
        if message_reasoning is not None:
            sample.messages[-1]["reasoning"] = message_reasoning or ""
            sample.messages[-1]["content"] = message_content
        else:
            if not self.reasoning_warning_issued:
                logging.warning(
                    "Reasoning content is not present in the assistant message, only including content. "
                    "Did you specify --enable-reasoning and --reasoning-parser [model_family] for vllm?"
                )
                self.reasoning_warning_issued = True

            sample.messages[-1]["content"] = message_content
            sample.messages[-1]["reasoning"] = ""

                
        return sample

# ---- Simplified write + concat ----

async def write_samples_to_disk(
    buffer: list[Sample],
    output_dir: str,
    chunk_count: int,
    input_schema: pl.Schema = None,   # kept for signature compatibility; not used
):
    output_path = os.path.join(output_dir, f"chunk_{chunk_count}.parquet")
    print(f"Writing {len(buffer)} samples to disk: {output_path}")

    # shortest/longest debug dumps (unchanged)
    l_min = float("inf")
    l_max = 0
    s_min = None
    s_max = None

    for s, sample in enumerate(buffer):
        l = sample.get_total_response_length()
        if l < l_min:
            l_min = l
            s_min = s
        if l > l_max:
            l_max = l
            s_max = s

    if s_min is not None and s_max is not None:
        with open(os.path.join(output_dir, f"shortest_chunk_{chunk_count}.txt"), "w") as f:
            f.write(buffer[s_min].to_readable_string())
        with open(os.path.join(output_dir, f"longest_chunk_{chunk_count}.txt"), "w") as f:
            f.write(buffer[s_max].to_readable_string())

    # rows -> json-serialize the problematic nested cols -> DF
    row_dicts = [sample.to_row_dict() for sample in buffer]
    for r in row_dicts:
        for c in SERIALIZED_COLS:
            if c in r:
                r[c] = _json_dumps_or_none(r[c])

    df = pl.DataFrame(row_dicts)

    # enforce only the generated column dtypes + serialized cols are Utf8
    df = df.with_columns([
        pl.col("answer").cast(pl.Utf8),
        pl.col("usage").cast(GENERATED_COLUMNS_SCHEMA["usage"]),
        *[pl.col(c).cast(pl.Utf8) for c in SERIALIZED_COLS if c in df.columns],
    ])

    df.write_parquet(output_path)


def concatenate_parquet_chunks(output_dir: str) -> list[str]:
    output_files = sorted(glob.glob(os.path.join(output_dir, "chunk_*.parquet")))
    if not output_files:
        print(f"No chunk parquet files found in {output_dir}")
        return []

    dfs = []
    for file in output_files:
        df = pl.read_parquet(file)
        print(
            f"Reading {os.path.basename(file)}: {len(df)} rows, "
            f"{len(df.columns)} columns: {df.columns}"
        )

        # hard-enforce: serialized cols are Utf8, generated cols are typed
        exprs = []
        for c in SERIALIZED_COLS:
            if c in df.columns:
                exprs.append(pl.col(c).cast(pl.Utf8))

        # keep these consistent too
        if "answer" in df.columns:
            exprs.append(pl.col("answer").cast(pl.Utf8))
        if "usage" in df.columns:
            exprs.append(pl.col("usage").cast(GENERATED_COLUMNS_SCHEMA["usage"]))

        if exprs:
            df = df.with_columns(exprs)

        dfs.append(df)

    out = pl.concat(dfs, how="diagonal_relaxed")
    output_path = os.path.join(output_dir, "all_samples.parquet")
    out.write_parquet(output_path)

    print(f"Concatenated {len(out)} samples into {output_path}")
    return output_files


async def flush_output_to_disk(input_stream: AsyncIterable[Sample], output_dir: str, chunk_size: int, input_schema: pl.Schema = None) -> AsyncIterable[Sample]:
    chunk_count = 0
    buffer = []
    async for sample in input_stream:
        buffer.append(sample)
        
        if len(buffer) >= chunk_size:
            await write_samples_to_disk(buffer, output_dir, chunk_count, input_schema)
            buffer = []
            chunk_count += 1
    
    if buffer:
        await write_samples_to_disk(buffer, output_dir, chunk_count, input_schema)


async def main(args):
    # if os.path.exists(args.output_dir):
    #     raise ValueError(f"Output directory {args.output_dir} already exists")
    
    # TODO: use data source to determine delims or expect none
    # delims = ("__","__")
    delims = None
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments as JSON to output directory
    args_dict = vars(args)
    args_path = os.path.join(args.output_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(args_dict, f, indent=2)
    print(f"Saved arguments to {args_path}")
    
    # load parquet file into polars dataframe
    dataset = pl.read_parquet(args.data_path)
    n_samples = len(dataset)

    # take head of dataset up to max_prompts
    if args.max_prompts is not None:
        # new length is min of max_prompts and original length
        n_samples = min(args.max_prompts, n_samples)
        dataset = dataset.head(n_samples)
        print(f"Taking head of dataset to {n_samples} prompts")
    else:
        print("No max_prompts specified, using full dataset")
    
    chunk_size = (n_samples*args.n) // 10

    print(f"Loaded {n_samples} rows from {args.data_path}")
    print(f"chunk size: {chunk_size} (accounting for {args.n} responses per prompt)")
    print(f"Columns: {dataset.columns}")
    
    # initialize inference manager
    inference_manager = InferenceManager(
        http_endpoint=args.endpoint, 
        api_key_env=args.api_key_env,
        model_name=args.model, 
        max_requests=args.max_requests, 
        max_attempts=args.infer_max_attempts, 
        base_delay=args.infer_base_delay, 
        max_delay=args.infer_max_delay, 
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p, 
        top_k=args.top_k,
        n=args.n)
    

    # define async generators for samples, inferred samples, and chunked disk flush generator
    async def samples_generator(dataset: pl.DataFrame) -> AsyncIterable[Sample]:
        for row in dataset.iter_rows(named=True):
            yield Sample(row)
    
    async def inferred_samples_generator(input_stream, inference_manager):
        tasks = []
        async for sample in input_stream:
            tasks.append(asyncio.create_task(inference_manager.infer(sample)))

        for t in asyncio.as_completed(tasks):  # yields tasks in completion order
            for out_sample in await t:         # infer() returns list[Sample]
                yield out_sample
        
    input_stream = samples_generator(dataset)
    inferred_stream = inferred_samples_generator(input_stream, inference_manager)
    
    # Pass input dataset schema to preserve column types
    await flush_output_to_disk(inferred_stream, args.output_dir, chunk_size, input_schema=dataset.schema)
    
    print(f"Finished evaluating {n_samples} samples")
    
    # Concatenate all chunk parquet files into a single parquet file
    chunk_files = concatenate_parquet_chunks(args.output_dir)
    
    if not chunk_files:
        return
    
    # # Delete all chunk parquet files (but keep all_samples.parquet)
    # for file in chunk_files:
    #     os.remove(file)
        
    # print(f"Deleted {len(chunk_files)} chunk parquet files in {args.output_dir}")
    
    return
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset locally as a parquet file. Must be in Verl or SkyRL convention")
    parser.add_argument("--model", type=str, required=True, help="HF model name")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--temperature", type=float, required=True, help="Temperature")
    parser.add_argument("--top-p", type=float, required=False, default=1.0, help="Top-p")
    parser.add_argument("--top-k", type=int, required=False, default=0, help="Top-k")
    parser.add_argument("--n", type=int, required=False, default=1, help="Number of responses per prompt")
    parser.add_argument("--max-tokens", type=int, required=False, default=1024, help="Max tokens generated per response")
    parser.add_argument("--seed", type=int, required=False, default=None, help="Seed")
    parser.add_argument("--max-prompts", type=int, required=False, default=None, help="Max number of prompts to evaluate")

    # optional vllm arg specifying http endpoint for vllm API, default None
    parser.add_argument("--endpoint", type=str, required=True, help="HTTP endpoint for API, one of openrouter, openai, or vllm." 
                        "If vllm, you MUST have initialized with --enable-reasoning and --reasoning-parser [model_family] appropriately for the model")
    
    # which api key env var to use for requests/authentication
    parser.add_argument("--api-key-env", type=str, required=True, help="Environment variable name for API key, e.g. OPENAI_API_KEY, OPENROUTER_API_KEY. For VLLM, you should set your own key and export to your desired env var.")
    
    # evaluation and inference configuration
    parser.add_argument("--n-workers", type=int, required=False, default=16, help="Number of worker processes for evaluation")
    parser.add_argument("--max-requests", type=int, required=False, default=16, help="Maximum number of concurrent requests to the inference engine")
    
    # inference retry configuration
    parser.add_argument("--infer-max-attempts", type=int, required=False, default=4, help="Maximum number of retry attempts for inference operations")
    parser.add_argument("--infer-base-delay", type=float, required=False, default=60.0, help="Base delay in seconds for exponential backoff in inference retries")
    parser.add_argument("--infer-max-delay", type=float, required=False, default=1200.0, help="Maximum delay in seconds for inference retries")
    
    # evaluation retry configuration
    parser.add_argument("--eval-max-attempts", type=int, required=False, default=4, help="Maximum number of retry attempts for evaluation operations")
    parser.add_argument("--eval-base-delay", type=float, required=False, default=16, help="Base delay in seconds for exponential backoff in evaluation retries")
    parser.add_argument("--eval-max-delay", type=float, required=False, default=128, help="Maximum delay in seconds for evaluation retries")

    args = parser.parse_args()
    
    asyncio.run(main(args))


"""
Example usage:

python3 recipe/zyphra/eval/eval_with_api.py \
--data-path /datasets/fldx2.parquet \
--model MiniMaxAI/MiniMax-M2 \
--output-dir test_api_gen \
--endpoint http://vllm:8000/v1/chat/completions \
--temperature 0.85 \
--top-p 0.95  \
--top-k 40 \
--n 4 \
--max-tokens 1024 \
--api-key-env VLLM_API_KEY \
--max-tokens 32768 \
--n-workers 64

"""
