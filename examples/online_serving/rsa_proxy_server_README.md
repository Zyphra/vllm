# RSA Chat Completion Proxy

`rsa_proxy_server.py` is an OpenAI-compatible proxy for running Recursive
Self-Aggregation (RSA) on top of a normal chat completions backend. It can sit
in front of `vllm serve` or any other OpenAI-compatible server.

Requests without RSA parameters pass through unchanged. Requests with RSA
parameters are expanded into multiple upstream `/v1/chat/completions` calls:
the proxy samples an initial population, optionally extracts or summarizes the
responses, builds aggregation prompts, and returns the final aggregated
response.

## Start The Proxy

Start an upstream vLLM server:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

Start the RSA proxy in front of it:

```bash
python examples/online_serving/rsa_proxy_server.py \
    --upstream-base-url http://localhost:8000/v1 \
    --port 8001
```

Send clients to the proxy instead of the upstream server:

```text
http://localhost:8001/v1
```

The proxy also forwards other paths to the upstream server, so model discovery
continues to work through `/v1/models`.

## Request Parameters

Add RSA parameters directly to the chat completion JSON body, or pass them as
`extra_body` when using the OpenAI Python client. The proxy removes these fields
before forwarding requests upstream.

Core RSA controls:

- `rsa_t`: Number of RSA refinement iterations. Set to `0` or omit RSA params
  for normal pass-through behavior.
- `rsa_n`: Initial population size, called `G` in the eval implementation.
- `rsa_k`: Number of candidate responses sampled into each aggregation prompt.
- `rsa_final_n`: Number of final aggregation prompts. `0` means use `rsa_n`.

Intermediate response controls:

- `rsa_summarize`: If true, summarize candidates between iterations. If false,
  extract reasoning directly from the generated responses.
- `short_resp_len`: Max tokens for non-final RSA generations and summaries.
- `rsa_tail_window`: If true, keep only the tail of extracted reasoning when
  `rsa_summarize` is false.
- `tail_window_len`: Approximate number of whitespace tokens to keep for the
  tail window.
- `use_pacore`: Use PaCoRe-style extraction and aggregation prompt wording.

Final response controls:

- `final_temperature`: Optional temperature override for the final RSA
  generation.
- `final_top_p`: Optional top-p override for the final RSA generation.
- `final_top_k`: Optional top-k override for the final RSA generation.

Debug and metadata controls:

- `return_aggregation_prompt`: If true, include one final aggregation prompt in
  the response as `aggregation_prompt`.
- `return_rsa_metadata`: If true, include `rsa_metadata`. Defaults to true.
- `summary_prompt`: Optional custom summarization instruction.

Aliases are accepted for compatibility with eval-style names, including
`rsa_tail_window_len`, `rsa_short_resp_len`, `rsa_use_pacore`,
`rsa_final_temperature`, `rsa_final_top_p`, `rsa_final_top_k`, and
`final_aggregation_prompts`.

RSA requests currently require `stream=false`. Non-RSA streaming requests pass
through to the upstream server.

## Python Client Examples

### Normal Pass-Through

This request does not include RSA parameters, so the proxy behaves like a normal
OpenAI-compatible server.

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")
model = client.models.list().data[0].id

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "Explain why the sky is blue in one paragraph."},
    ],
)
print(response.choices[0].message.content)
```

### Basic RSA For Hard Reasoning

This samples eight initial solutions, aggregates four at a time for two RSA
iterations, and returns one final answer.

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")
model = client.models.list().data[0].id

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Solve: If x^2 - 5x + 6 = 0, what are the possible values of x?",
        },
    ],
    temperature=0.8,
    max_tokens=4096,
    extra_body={
        "rsa_t": 2,
        "rsa_n": 8,
        "rsa_k": 4,
        "short_resp_len": 2048,
        "rsa_final_n": 1,
        "final_temperature": 0.2,
        "return_aggregation_prompt": True,
    },
)

print(response.choices[0].message.content)
print(response.model_extra["aggregation_prompt"])
print(response.model_extra["rsa_metadata"]["rsa_token_count_history"])
```

### RSA With Summaries

Use summarization when intermediate responses are long and you want each next
iteration to aggregate compact candidate summaries instead of full reasoning.

```python
response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Work through this proof problem carefully: ...",
        },
    ],
    max_tokens=8192,
    extra_body={
        "rsa_t": 2,
        "rsa_n": 6,
        "rsa_k": 3,
        "rsa_summarize": True,
        "short_resp_len": 2048,
        "rsa_final_n": 2,
        "final_temperature": 0.4,
    },
)
```

### PaCoRe-Style Aggregation

Use `use_pacore` when you want the aggregation prompt to treat prior responses
as references rather than candidate math solutions.

```python
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "Analyze this design tradeoff: ..."},
    ],
    extra_body={
        "rsa_t": 1,
        "rsa_n": 5,
        "rsa_k": 5,
        "use_pacore": True,
        "tail_window_len": 1200,
        "return_aggregation_prompt": True,
    },
)
```

## Curl Example

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Solve the problem and explain your reasoning: 15 * 17"}
    ],
    "temperature": 0.8,
    "max_tokens": 2048,
    "rsa_t": 1,
    "rsa_n": 4,
    "rsa_k": 2,
    "rsa_final_n": 1,
    "final_temperature": 0.2,
    "return_aggregation_prompt": true
  }'
```

## Response Shape

The response is still an OpenAI-style chat completion object. For RSA requests,
the proxy replaces `choices` with the final RSA generation choices and adds
aggregate usage across all upstream calls.

When `return_rsa_metadata` is enabled, the response includes:

- `rsa_metadata.rsa_iterations`: Number of refinement iterations.
- `rsa_metadata.rsa_n`: Initial population size.
- `rsa_metadata.rsa_k`: Aggregation sample size.
- `rsa_metadata.rsa_token_count_history`: Mean output token count per
  iteration.

When `return_aggregation_prompt` is enabled, the response also includes a single
`aggregation_prompt`. If `rsa_final_n` is greater than one, this is the first
final aggregation prompt.

## Operational Notes

- Run the proxy as a separate process from `vllm serve`.
- Point clients at the proxy port, not the upstream server port.
- Keep `rsa_n`, `rsa_t`, and `rsa_final_n` modest at first. The number of
  upstream calls grows with the population size and number of iterations.
- Use `rsa_summarize=true` for very long intermediate generations.
- Use `rsa_tail_window=true` and tune `tail_window_len` when summaries are not
  needed but full intermediate reasoning is too long.
- The proxy forwards request `Authorization` headers to the upstream server. If
  a request does not include one, `--api-key` or `OPENAI_API_KEY` is used.
