"""Run with torchrun --standalone --nproc-per-node=2 (or 3 --subgroup).

Real FSDP backward, clipping, native-Muon comparison, freezing and rank-local
resume. No model downloads. Temporary checkpoint files stay under tests/.
"""

import argparse
import copy
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from litespec.training.checkpoint import (
    load_distributed_training_state,
    save_distributed_training_state,
)
from litespec.training.fsdp_layout import build_fsdp_layouts
from litespec.training.optimizer import (
    BF16Optimizer,
    CosineAnnealingWarmupLR,
    normalize_accumulated_gradients,
    register_optimizer_parameters,
)


class TinyDraft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(7, 5) for _ in range(7)])
        self.fc = torch.nn.Linear(3, 9, bias=False)
        self.norm = torch.nn.LayerNorm(5)
        self.layer_depth_embedding = torch.nn.Embedding(11, 5)
        self.sequential_head = torch.nn.Linear(5, 5)
        self.unused = torch.nn.Linear(2, 2, bias=False)

    def forward(self, scale):
        loss = scale.new_zeros((), dtype=torch.float32)
        for name, p in self.named_parameters():
            if name.startswith("unused."):
                continue
            loss = (
                loss
                + (p.float().square().sum() + p.float().sum() * 0.17) * scale.float()
            )
        return loss


def run_case(group, matrices_per_rank, checkpoint_dir):
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    torch.manual_seed(47)
    model = TinyDraft().cuda().bfloat16()
    parameters, specs = register_optimizer_parameters(model)
    refs = [torch.nn.Parameter(p.detach().float().clone()) for p in parameters]
    fsdp = FSDP(
        model,
        process_group=group,
        use_orig_params=True,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
        ),
    )
    layouts = build_fsdp_layouts(fsdp, parameters, specs, group)
    config = {"matrices_per_rank": matrices_per_rank}

    def make_optimizer():
        return BF16Optimizer(
            model,
            lr=0.7,
            parameters=parameters,
            parameter_specs=specs,
            parameter_layouts=layouts,
            process_group=group,
            muon_config=config,
            max_grad_norm=1,
            total_steps=8,
            warmup_ratio=0.25,
        )

    opt = make_optimizer()
    native_muon = torch.optim.Muon(
        [p for p, s in zip(refs, specs) if s.optimizer == "muon"],
        lr=0.02,
        momentum=0.95,
        weight_decay=0.1,
    )
    native_adam = torch.optim.AdamW(
        [p for p, s in zip(refs, specs) if s.optimizer == "adamw"],
        lr=0.0004,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    )
    schedulers = [
        CosineAnnealingWarmupLR(o, total_steps=8, warmup_steps=2)
        for o in (native_muon, native_adam)
    ]
    max_error = 0.0
    for step in range(5):
        # Freeze/unfreeze in both directions without changing registration.
        model.sequential_head.requires_grad_(step >= 2)
        model.layers[0].requires_grad_(step < 3)
        for micro in range(2):
            loss = fsdp(torch.tensor(rank + micro + 1.0, device="cuda"))
            (loss / 2).backward()
        normalize_accumulated_gradients(opt, torch.tensor(2.0, device="cuda"), 2, group)
        local_grads = [
            p.grad.detach().float().cpu() if p.grad is not None else None
            for p in parameters
        ]
        gathered = [None] * world
        dist.all_gather_object(gathered, local_grads, group=group)
        for i, (ref, layout) in enumerate(zip(refs, layouts)):
            enabled = any(grads[i] is not None for grads in gathered)
            if not enabled:
                ref.grad = None
                continue
            gradient = torch.zeros_like(ref).flatten()
            for peer, grads in enumerate(gathered):
                offset, length = layout.shards[peer]
                if length and grads[i] is not None:
                    gradient[offset : offset + length].copy_(grads[i].flatten())
            ref.grad = gradient.reshape(ref.shape)
        reference_norm = torch.nn.utils.clip_grad_norm_(refs, 1)
        actual_norm = opt.step()
        assert abs(actual_norm - reference_norm.item()) < 1e-4
        native_muon.step()
        native_adam.step()
        for scheduler in schedulers:
            scheduler.step()
        for i, (master, ref, layout) in enumerate(zip(opt.fp32_params, refs, layouts)):
            offset, length = layout.shards[rank]
            expected = ref.flatten()[offset : offset + length]
            if length:
                max_error = max(
                    max_error, (master.flatten() - expected).abs().max().item()
                )
            torch.testing.assert_close(master.flatten(), expected, rtol=0, atol=1e-3)
            torch.testing.assert_close(
                parameters[i], master.to(parameters[i].dtype), rtol=0, atol=0
            )
            if length and layout.spec.optimizer == "muon" and ref in native_muon.state:
                torch.testing.assert_close(
                    opt.muon_optimizer.state[master]["momentum_buffer"].flatten(),
                    native_muon.state[ref]["momentum_buffer"].flatten()[
                        offset : offset + length
                    ],
                    rtol=1e-5,
                    atol=1e-7,
                )
        if step == 2:
            saved = copy.deepcopy(opt.state_dict())
            save_distributed_training_state(checkpoint_dir, saved, process_group=group)
            restored = load_distributed_training_state(
                checkpoint_dir, process_group=group
            )
            before = [p.clone() for p in opt.fp32_params]
            opt = make_optimizer()
            opt.load_state_dict(restored)
            for actual, expected in zip(opt.fp32_params, before):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    maximum = torch.tensor(max_error, device="cuda")
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    if rank == 0:
        print(
            f"PASS world={world} matrices_per_rank={matrices_per_rank} max_master_error={maximum.item():.8f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subgroup",
        action="store_true",
        help="Exclude global rank zero; launch at least 3 ranks",
    )
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    ranks = list(range(1 if args.subgroup else 0, dist.get_world_size()))
    if not ranks or (args.subgroup and len(ranks) < 2):
        raise ValueError("--subgroup requires at least three ranks")
    group = dist.new_group(ranks) if args.subgroup else dist.group.WORLD
    directory = (
        tempfile.TemporaryDirectory(prefix="muon-smoke-", dir=Path(__file__).parent)
        if dist.get_rank() == ranks[0]
        else None
    )
    paths = [directory.name if directory else None]
    dist.broadcast_object_list(paths, src=ranks[0])
    try:
        if dist.get_rank() in ranks:
            for batch_size in (1, 2):
                run_case(group, batch_size, paths[0])
        dist.barrier()
    finally:
        if directory:
            directory.cleanup()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
