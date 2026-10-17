"""Two-rank exact native reductions versus an unsharded full-window reference."""

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from test_implementation import config
from litespec.modeling.draft.composable import ComposableDraftModel
from litespec.core.composable.training_model import ComposableTrainingModel
from litespec.core.composable.batch import BlockLayout
from litespec.training.recipes.composable import ADAPTER
from litespec.training.runner import OptimizerWindow

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
root = Path(__file__).resolve().parent
records = []
for objective in ("hard_ce", "nla", "dpace"):
    for count, empty_rank in ((2, False), (1, False), (1, True)):
        torch.manual_seed(81)
        model = ComposableDraftModel(
            config("dflare", "selector", objective, heterogeneous=True, window=True)
        ).cuda()
        reference = copy.deepcopy(model)
        embedding = nn.Embedding(23, 32).cuda().requires_grad_(False)
        head = nn.Linear(32, 23, bias=False).cuda().requires_grad_(False)
        wrapper = ComposableTrainingModel(
            model, head, embedding, 0, num_anchors=2, objective_chunk_blocks=1
        )
        fsdp = FSDP(
            wrapper,
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            device_id=rank,
            ignored_modules=[embedding, head],
        )
        generator = torch.Generator().manual_seed(100 + rank)
        data = []
        for index in range(count):
            n = 1 + rank
            ids = torch.randint(23, (n, 10), generator=generator)
            mask = torch.ones(n, 10)
            mask[:, -(rank + index + 1) :] = 0
            if empty_rank and rank == 1:
                mask.zero_()
            data.append(
                {
                    "input_ids": ids,
                    "loss_mask": mask,
                    "hidden": torch.randn(n, 10, 64, generator=generator),
                    "teacher": torch.randn(n, 10, 32, generator=generator),
                }
            )
        session = SimpleNamespace(
            draft=model,
            topology=None,
            tp_draft_rank=None,
            process_group=None,
            args=SimpleNamespace(
                num_anchors=2,
                window_buffer_mb=1,
                cache_dir=str(root / "gradient_cache"),
            ),
        )
        torch.manual_seed(400 + rank)
        prepared = ADAPTER.prepare_optimizer_window(
            session, data[0], iter(data[1:]), count
        )
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(
            gathered, [(item.data, item.layout) for item in prepared]
        )
        all_rows = [row for group in gathered for row in group]

        class GradientSink:
            def scale_model_gradients(self, scale):
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)

            def step(self):
                return 0.0

        window = OptimizerWindow(GradientSink(), 2)
        for item in prepared:
            raw = item.materialize()
            output = fsdp(
                raw["input_ids"].cuda(),
                raw["hidden"].cuda(),
                raw["loss_mask"].cuda(),
                raw["teacher"].cuda(),
                layout=item.layout.to("cuda"),
                window_denominators=item.window_denominators,
                selector_loss_alpha=0.7,
            )
            window.backward(output)
        window.step()
        reference_wrapper = ComposableTrainingModel(
            reference, head, embedding, 0, num_anchors=2, objective_chunk_blocks=1
        )
        full_layout = BlockLayout(
            **{
                key: torch.cat([getattr(layout, key) for _, layout in all_rows]).cuda()
                for key in prepared[0].layout.__dataclass_fields__
            }
        )
        output = reference_wrapper(
            torch.cat([raw["input_ids"] for raw, _ in all_rows]).cuda(),
            torch.cat([raw["hidden"] for raw, _ in all_rows]).cuda(),
            torch.cat([raw["loss_mask"] for raw, _ in all_rows]).cuda(),
            torch.cat([raw["teacher"] for raw, _ in all_rows]).cuda(),
            layout=full_layout,
            selector_loss_alpha=0.7,
        )
        output.loss.backward()
        errors = []
        with FSDP.summon_full_params(fsdp, with_grads=True):
            for (name, p), (_, q) in zip(
                model.named_parameters(), reference.named_parameters()
            ):
                torch.testing.assert_close(
                    p.grad, q.grad, atol=1e-5, rtol=2e-4, msg=name
                )
                errors.append((p.grad - q.grad).abs().max().item())
        records.append(
            {
                "objective": objective,
                "logical_batches": count,
                "one_empty_rank": empty_rank,
                "max_gradient_error": max(errors),
            }
        )
        for item in prepared:
            item.release()
        del fsdp, wrapper, model, reference, reference_wrapper, window, output
        dist.barrier()
if rank == 0:
    (root / "distributed_gradients_result.json").write_text(
        json.dumps(records, indent=2)
    )
    print("DISTRIBUTED_GRADIENTS_OK", records, flush=True)
dist.destroy_process_group()
