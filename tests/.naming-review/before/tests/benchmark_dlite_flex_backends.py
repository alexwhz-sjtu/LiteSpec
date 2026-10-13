"""Manual DLite FlexAttention backend benchmark.

Example (PyTorch >= 2.11 with an NVIDIA GPU)::

    python tests/benchmark_dlite_flex_backends.py --width 78 --history-length 4096

The reported throughput uses the same number of *real* query tokens for both
backends. FLASH may execute extra padded blocks; TRITON always uses the natural
shape.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from litespec.core.attention_masks import (
    create_dlite_flex_mask,
    flash_kv_padding,
    flash_padded_block_width,
)
from litespec.modeling.draft.flex_attention import (
    compile_friendly_flex_attention,
    flex_kernel_options,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=78)
    parser.add_argument("--query-length", type=int, default=8)
    parser.add_argument("--chs-slots", type=int, default=14)
    parser.add_argument("--history-length", type=int, default=0)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def benchmark(args, backend: str):
    device = torch.device("cuda")
    width = (
        flash_padded_block_width(args.width, args.query_length, args.chs_slots)
        if backend == "FLASH"
        else args.width
    )
    q_len = width * args.query_length
    kv_len = args.history_length + width * (args.chs_slots + args.query_length)
    if backend == "FLASH":
        kv_len += flash_kv_padding(kv_len)

    anchors = torch.arange(width, device=device).view(1, -1)
    keep = torch.arange(width, device=device).view(1, -1) < args.width
    token_keep = torch.ones(
        (1, width, args.query_length), dtype=torch.bool, device=device
    )
    physical_history = kv_len - width * (args.chs_slots + args.query_length)
    mask = create_dlite_flex_mask(
        model_role="swa_teacher" if physical_history else "student",
        anchor_positions=anchors,
        block_keep_mask=keep,
        token_keep_mask=token_keep,
        seq_len=physical_history,
        swa_window_size=max(args.history_length, 1),
        chs_slots=args.chs_slots,
        query_len=args.query_length,
        device=device,
        flex_backend=backend,
    )
    q = torch.randn(
        1,
        args.heads,
        q_len,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn(
        1,
        args.kv_heads,
        kv_len,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    v = torch.randn_like(k, requires_grad=True)

    def step():
        output = compile_friendly_flex_attention(
            q,
            k,
            v,
            block_mask=mask,
            enable_gqa=args.heads != args.kv_heads,
            kernel_options=flex_kernel_options(backend),
        )
        # Match training: padded query rows have zero objective weight.
        output[
            :, :, : args.width * args.query_length
        ].float().square().mean().backward()
        q.grad = k.grad = v.grad = None

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)

    latency_ms = statistics.median(samples)
    real_tokens = args.width * args.query_length
    return width, q_len, kv_len, latency_ms, real_tokens / (latency_ms / 1000.0)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    print("backend physical_width q_len kv_len median_ms real_query_tokens/s")
    for backend in ("TRITON", "FLASH"):
        width, q_len, kv_len, latency, throughput = benchmark(args, backend)
        print(
            f"{backend:7s} {width:14d} {q_len:5d} {kv_len:6d} "
            f"{latency:9.3f} {throughput:19.0f}"
        )


if __name__ == "__main__":
    main()
