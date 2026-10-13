"""Warm CUDA forward+backward benchmark; results are JSON for later comparison.

PYTHONPATH=. .venv/bin/python tests/benchmark_loss_fusion.py --rows 512 --vocab 151936
"""

import argparse
import gc
import json
import os
import statistics
from pathlib import Path

import torch

from litespec.core.losses import (
    weighted_cross_entropy,
    weighted_ce_l1,
    weighted_kl,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=32768)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--note",
        default="GPU exclusivity not verified; check contention before comparing timings.",
    )
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    torch.manual_seed(17)
    dtype = getattr(torch, args.dtype)
    x = torch.randn(
        args.rows, args.vocab, device="cuda", dtype=dtype, requires_grad=True
    )
    target = torch.randn_like(x)
    labels = torch.randint(args.vocab, (args.rows,), device="cuda")
    weights = torch.rand(args.rows, device="cuda")
    results = {}
    for kind in ("ce", "ce_l1", "kl"):
        steps = {}
        for backend in ("torch", "inductor"):
            os.environ["LITESPEC_LOSS_BACKEND"] = backend

            def step():
                x.grad = None
                if kind == "ce":
                    loss = weighted_cross_entropy(x, labels, weights)[0]
                elif kind == "ce_l1":
                    ce, l1, *_ = weighted_ce_l1(x, labels, target, weights)
                    loss = 0.1 * ce + l1
                else:
                    loss = weighted_kl(x, target, weights)
                loss.backward()

            for _ in range(3):
                step()
            torch.cuda.synchronize()
            steps[backend] = step
        times = {name: [] for name in steps}
        peaks = {name: [] for name in steps}
        for trial in range(args.trials):
            for backend in (
                ("torch", "inductor") if trial % 2 == 0 else ("inductor", "torch")
            ):
                os.environ["LITESPEC_LOSS_BACKEND"] = backend
                x.grad = None
                gc.collect()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                for _ in range(args.iterations):
                    steps[backend]()
                end.record()
                end.synchronize()
                times[backend].append(start.elapsed_time(end) / args.iterations)
                peaks[backend].append(
                    (torch.cuda.max_memory_allocated() - base) / 2**20
                )
        for backend in steps:
            result = {
                "ms": statistics.median(times[backend]),
                "trials_ms": times[backend],
                "extra_peak_mib": max(peaks[backend]),
            }
            results[f"{kind}/{backend}"] = result
            print(json.dumps({f"{kind}/{backend}": result}), flush=True)
    report = {
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "results": results,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
