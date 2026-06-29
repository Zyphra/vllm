# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
An example shows how to generate chat completions from Markovian RSA reasoning models
like Zyphra/ZAYA1-8B.

To run this example, you need to start the vLLM server
with the Markovian RSA proxy:

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
Then prompt the server with RSA flags

This example demonstrates how to generate chat completions with Markovian RSA
using the OpenAI Python client library.
"""

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"


def main():
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    # 2026 AIME I Problem 14
    messages = [{"role": "user", "content": r"In an equiangular pentagon, the sum of the squares of the side lengths equals $308$, and the sum of the squares of the diagonal lengths equals $800$. The square of the perimeter of the pentagon can be expressed as $m\sqrt{n}$, where $m$ and $n$ are positive integers and $n$ is not divisible by the square of any prime. Find $m+n$."}]
    response = client.chat.completions.create(
        model=model, 
        messages=messages,
        temperature=1.0,
        max_tokens=61440,
        extra_body={
            "rsa_t": 8,
            "rsa_n": 32,
            "rsa_k": 4,
            "short_resp_len": 32768,
            "rsa_tail_window": True,
            "tail_window_len": 4096,
            "rsa_final_n": 1,
            "final_temperature": 0.6,
            "return_aggregation_prompt": True,
            "rsa_debug": True,
        })

    reasoning = response.choices[0].message.reasoning
    content = response.choices[0].message.content

    print("reasoning", reasoning)
    print("content", content)


if __name__ == "__main__":
    main()
