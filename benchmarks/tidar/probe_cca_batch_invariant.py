# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import time

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops import cca_conv1d_batch_invariant


def tensor_hash(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def reference_conv(
    x: torch.Tensor,
    dw_weight: torch.Tensor,
    dw_bias: torch.Tensor,
    gw_weight: torch.Tensor,
    gw_bias: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    mid = F.conv1d(
        x.float(), dw_weight.float(), dw_bias.float(), groups=x.shape[1])
    return F.conv1d(
        mid, gw_weight.float(), gw_bias.float(), groups=groups)


def timed_ms(fn, repeats: int) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--large-batch", type=int, default=64)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=19)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    dtype = torch.bfloat16
    channels = args.groups * args.head_dim
    x1 = torch.randn(
        1, channels, args.seq_len, device=device, dtype=dtype)
    x_large = torch.randn(
        args.large_batch,
        channels,
        args.seq_len,
        device=device,
        dtype=dtype,
    )
    x_large[0].copy_(x1[0])
    dw_weight = torch.randn(
        channels, 1, 2, device=device, dtype=dtype) * 0.02
    dw_bias = torch.randn(channels, device=device, dtype=dtype) * 0.02
    gw_weight = torch.randn(
        channels, args.head_dim, 2, device=device, dtype=dtype) * 0.02
    gw_bias = torch.randn(channels, device=device, dtype=dtype) * 0.02

    def run(x: torch.Tensor) -> torch.Tensor:
        return cca_conv1d_batch_invariant(
            x,
            dw_weight,
            dw_bias,
            gw_weight,
            gw_bias,
            args.groups,
        )

    def run_ref(x: torch.Tensor) -> torch.Tensor:
        return reference_conv(
            x, dw_weight, dw_bias, gw_weight, gw_bias, args.groups)

    out1 = run(x1)
    out_large = run(x_large)
    ref1 = run_ref(x1)
    shape_equal = torch.equal(out1[0], out_large[0])
    max_abs = (out1.float() - ref1.float()).abs().max().item()
    mismatch = torch.count_nonzero(out1 != ref1).item()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run(x_large)
    graph.replay()
    torch.cuda.synchronize()
    graph_equal = torch.equal(out_large, graph_out)

    print(
        "CCA_PROBE "
        f"device={torch.cuda.get_device_name(device)} "
        f"b1_hash={tensor_hash(out1[0])} "
        f"b{args.large_batch}_r0_hash={tensor_hash(out_large[0])} "
        f"exact={shape_equal} graph_exact={graph_equal} "
        f"ref_mismatch={mismatch}/{out1.numel()} "
        f"ref_max_abs={max_abs:.8f} "
        f"b1_ms={timed_ms(lambda: run(x1), args.repeats):.4f} "
        f"b{args.large_batch}_ms="
        f"{timed_ms(lambda: run(x_large), args.repeats):.4f} "
        f"ref_b1_ms={timed_ms(lambda: run_ref(x1), args.repeats):.4f} "
        f"ref_b{args.large_batch}_ms="
        f"{timed_ms(lambda: run_ref(x_large), args.repeats):.4f}"
    )
    if not shape_equal:
        raise SystemExit("batch-shape invariance check failed")
    if not graph_equal:
        raise SystemExit("cudagraph replay check failed")


if __name__ == "__main__":
    main()
