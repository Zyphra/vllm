# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import time

import torch
import torch.nn.functional as F


def tensor_hash(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def timed_ms(fn, repeats: int) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--small-rows", type=int, default=17)
    parser.add_argument("--large-rows", type=int, default=1088)
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    x_small = torch.randn(
        args.small_rows, args.in_features, device=device, dtype=dtype)
    x_large = torch.randn(
        args.large_rows, args.in_features, device=device, dtype=dtype)
    x_large[:args.small_rows].copy_(x_small)
    weight = torch.randn(
        args.out_features, args.in_features, device=device, dtype=dtype) * 0.02

    def run(x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, weight)

    out_small = run(x_small)
    out_large = run(x_large)
    large_prefix = out_large[:args.small_rows]
    exact = torch.equal(out_small, large_prefix)
    mismatches = torch.count_nonzero(out_small != large_prefix).item()
    max_abs = (out_small.float() - large_prefix.float()).abs().max().item()
    top1_equal = torch.equal(
        out_small.argmax(dim=-1), large_prefix.argmax(dim=-1))

    print(
        "GEMM_PROBE "
        f"label={args.label} device={torch.cuda.get_device_name(device)} "
        f"small_hash={tensor_hash(out_small)} "
        f"large_prefix_hash={tensor_hash(large_prefix)} "
        f"exact={exact} mismatches={mismatches}/{out_small.numel()} "
        f"max_abs={max_abs:.8f} top1_exact={top1_equal} "
        f"small_ms={timed_ms(lambda: run(x_small), args.repeats):.4f} "
        f"large_ms={timed_ms(lambda: run(x_large), args.repeats):.4f}"
    )


if __name__ == "__main__":
    main()
