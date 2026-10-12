"""torchrun benchmark of serial-matrix versus batched FSDP1 Muon updates.

Optimizer-only timing, fixed identical gradients/initial state, no downloads.
The serial reference uses the same routing/math but schedules one matrix at a
time, so batching is the only algorithmic difference.
"""

import argparse
import copy
import json
import os
import time
import types

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from litespec.training.fsdp_layout import build_fsdp_layouts
from litespec.training.optimizer import BF16Optimizer, register_optimizer_parameters


class MatrixBank(torch.nn.Module):
    def __init__(self, count, rows, cols):
        super().__init__()
        self.weights = torch.nn.ModuleList(
            [torch.nn.Linear(cols, rows, bias=False) for _ in range(count)]
        )

    def forward(self, inputs):
        return sum(layer(inputs).float().sum() for layer in self.weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--matrices", type=int, default=16)
    parser.add_argument("--matrices-per-rank", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if min(vars(args).values()) <= 0:
        parser.error("all arguments must be positive")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        torch.manual_seed(101)
        model = MatrixBank(args.matrices, args.rows, args.cols).cuda().bfloat16()
        parameters, specs = register_optimizer_parameters(model)
        fsdp_model = FSDP(
            model,
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        )
        layouts = build_fsdp_layouts(fsdp_model, parameters, specs)
        opt = BF16Optimizer(
            model,
            lr=0.1,
            parameters=parameters,
            parameter_specs=specs,
            parameter_layouts=layouts,
            muon_config={"matrices_per_rank": args.matrices_per_rank},
            total_steps=100000,
            warmup_ratio=0,
        )
        initial = copy.deepcopy(opt.state_dict())
        grads = [torch.randn_like(p) for p in parameters]
        original_batches = opt.muon_optimizer._batches

        def serial_batches(self, active):
            for i, enabled in enumerate(active):
                if enabled:
                    yield [i], [i % self.world]

        results = []
        for mode in ("serial_matrix", "batched"):
            opt.load_state_dict(copy.deepcopy(initial))
            opt.muon_optimizer._batches = (
                types.MethodType(serial_batches, opt.muon_optimizer)
                if mode == "serial_matrix"
                else original_batches
            )
            for _ in range(args.warmup):
                for p, g in zip(parameters, grads):
                    p.grad = g.clone()
                opt.step()
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(args.steps):
                for p, g in zip(parameters, grads):
                    p.grad = g.clone()
                opt.step()
            torch.cuda.synchronize()
            stats = torch.tensor(
                [
                    (time.perf_counter() - start) * 1000 / args.steps,
                    torch.cuda.max_memory_allocated() / 2**20,
                    opt.muon_optimizer.communication_calls,
                ],
                device="cuda",
                dtype=torch.float64,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.MAX)
            results.append(
                dict(
                    mode=mode,
                    step_ms=stats[0].item(),
                    peak_allocated_mib=stats[1].item(),
                    matrix_exchange_calls=int(stats[2].item()),
                )
            )
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "world_size": dist.get_world_size(),
                        "config": vars(args),
                        "results": results,
                    },
                    indent=2,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
