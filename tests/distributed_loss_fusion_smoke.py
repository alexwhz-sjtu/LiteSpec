"""Two-rank FSDP, weighted accumulation and optimizer-resume loss parity.

PYTHONPATH=. torchrun --standalone --nproc-per-node=2 tests/distributed_loss_fusion_smoke.py
"""

import copy
import os
from pathlib import Path

os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(
    Path(__file__).parent / ".cache" / "distributed-loss" / os.environ.get("RANK", "0")
)
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

from litespec.core.losses import weighted_ce_l1, weighted_kl
from litespec.core.outputs import RatioTerms, TrainingOutput
from litespec.training.optimizer import BF16Optimizer
from litespec.training.runner import OptimizerWindow


class LossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(16, 513)

    def forward(self, hidden, teacher, labels, weights):
        logits = self.head(hidden).float()
        ce, l1, *_ = weighted_ce_l1(logits, labels, teacher, weights)
        kl = weighted_kl(logits, teacher, weights)
        return 0.1 * ce + l1 + 0.7 * kl


def run(backend):
    os.environ["LITESPEC_LOSS_BACKEND"] = backend
    torch.manual_seed(91)
    model = LossModel().cuda().bfloat16()
    parameters = list(model.parameters())
    wrapped = FSDP(
        model, use_orig_params=True, sharding_strategy=ShardingStrategy.SHARD_GRAD_OP
    )

    def optimizer():
        return BF16Optimizer(
            model, lr=0.0004, parameters=parameters, total_steps=5, warmup_ratio=0
        )

    opt = optimizer()
    for step in range(3):
        window = OptimizerWindow(opt, accumulation_steps=2)
        for micro in range(2):
            torch.manual_seed(100 + step * 20 + micro * 3 + dist.get_rank())
            rows = 3 + dist.get_rank()
            hidden = torch.randn(rows, 16, device="cuda", dtype=torch.bfloat16)
            teacher = torch.randn(rows, 513, device="cuda")
            labels = torch.randint(513, (rows,), device="cuda")
            weights = torch.rand(rows, device="cuda")
            weights[0] = 0
            numerator = wrapped(hidden, teacher, labels, weights)
            denominator = weights.sum()
            window.backward(
                TrainingOutput(
                    numerator / denominator, RatioTerms(numerator, denominator)
                )
            )
        window.step()
        if step == 1:
            saved = copy.deepcopy(opt.state_dict())
            opt = optimizer()
            opt.load_state_dict(saved)
    result = [p.detach().clone() for p in opt.fp32_params]
    del window, opt, wrapped, model
    return result


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        reference = run("torch")
        actual = run("inductor")
        maximum = 0.0
        for a, b in zip(actual, reference):
            torch.testing.assert_close(a, b, rtol=5e-5, atol=1e-5)
            if a.numel():
                maximum = max(maximum, (a - b).abs().max().item())
        error = torch.tensor(maximum, device="cuda")
        dist.all_reduce(error, op=dist.ReduceOp.MAX)
        if dist.get_rank() == 0:
            print(
                f"PASS FSDP loss fusion + accumulation + resume: max master error={error.item():.8g}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
