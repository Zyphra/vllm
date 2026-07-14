# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the deterministic TiDAR live-training workload."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-groups", type=int, default=16)
    parser.add_argument("--responses-per-prompt", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=32768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_parquet(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rng = np.random.default_rng(args.seed)

    selected: list[dict[str, object]] = []
    for dataset_index in rng.permutation(len(data)):
        row = data.iloc[int(dataset_index)]
        messages = row["prompt"]
        if hasattr(messages, "tolist"):
            messages = messages.tolist()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        if len(prompt_ids) > args.max_prompt_tokens:
            continue
        if tokenizer.bos_token_id is not None:
            assert prompt_ids.count(tokenizer.bos_token_id) == 1
        selected.append(
            {
                "dataset_index": int(dataset_index),
                "data_source": str(row["data_source"]),
                "prompt_tokens": len(prompt_ids),
                "prompt": prompt,
            }
        )
        if len(selected) == args.prompt_groups:
            break

    if len(selected) != args.prompt_groups:
        raise RuntimeError(
            f"Selected {len(selected)} prompt groups; expected {args.prompt_groups}."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output_file:
        for sample_index in range(args.responses_per_prompt):
            for group_index, item in enumerate(selected):
                output_file.write(
                    json.dumps(
                        {
                            "prompt": item["prompt"],
                            "output_tokens": args.output_tokens,
                            "group_index": group_index,
                            "sample_index": sample_index,
                            "dataset_index": item["dataset_index"],
                            "data_source": item["data_source"],
                            "prompt_tokens": item["prompt_tokens"],
                        }
                    )
                    + "\n"
                )

    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = {
        "model": str(args.model),
        "dataset": str(args.dataset),
        "seed": args.seed,
        "enable_thinking": True,
        "max_prompt_tokens": args.max_prompt_tokens,
        "prompt_groups": args.prompt_groups,
        "responses_per_prompt": args.responses_per_prompt,
        "total_requests": args.prompt_groups * args.responses_per_prompt,
        "selection": [
            {key: value for key, value in item.items() if key != "prompt"}
            for item in selected
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
