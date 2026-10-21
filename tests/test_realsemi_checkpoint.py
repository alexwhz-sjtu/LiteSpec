"""Two-rank FSDP/Muon save-and-resume integration using a small RealSemi."""

import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litespec.training.runner import configure_rank_caches

configure_rank_caches()
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from test_realsemi import config
from litespec.modeling.draft.realsemi import RealSemiDraftModel
from litespec.core.realsemi import OnlineRealSemiModel
from litespec.training.setup import wrap_training_model
from litespec.training.optimizer import BF16Optimizer, register_optimizer_parameters
from litespec.training.fsdp_layout import build_fsdp_layouts
from litespec.training.runner import OptimizerWindow
from litespec.training.checkpoint import (
    save_checkpoint,
    load_distributed_training_state,
)
from litespec.training.algorithms import RunState, Phase

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
torch.set_num_threads(2)
dist.init_process_group("nccl")
torch.manual_seed(42)
model = RealSemiDraftModel(config("last", n=3, m=1)).cuda().bfloat16()
embed, head = (
    nn.Embedding(64, 32).cuda().bfloat16(),
    nn.Linear(32, 64, bias=False).cuda().bfloat16(),
)
online = OnlineRealSemiModel(
    model,
    head,
    embed,
    63,
    block_size=4,
    attention_backend="sdpa",
    num_anchors=2,
    objective_chunk_blocks=1,
)
parameters, specs = register_optimizer_parameters(model)
fsdp = wrap_training_model(online, SimpleNamespace(embed_tokens=embed, lm_head=head))
layouts = build_fsdp_layouts(fsdp, parameters, specs)
optimizer = BF16Optimizer(
    model,
    lr=0.0006,
    total_steps=4,
    warmup_ratio=0.0,
    parameters=parameters,
    parameter_specs=specs,
    parameter_layouts=layouts,
    muon_config={},
)
ids = torch.randint(0, 60, (1, 12), device="cuda")
features = torch.randn(1, 12, 64, device="cuda", dtype=torch.bfloat16)
teacher = torch.randn(1, 12, 32, device="cuda", dtype=torch.bfloat16)
output = fsdp(
    input_ids=ids,
    hidden_states=features,
    loss_mask=torch.ones(1, 12, device="cuda"),
    target_last_hidden_states=teacher,
)
window = OptimizerWindow(optimizer, 1)
window.backward(output)
norm = window.step()
assert torch.isfinite(torch.as_tensor(norm)).all()
metadata = dict(
    cursor=dataclasses.asdict(RunState(global_step=1, optimizer_step=1)),
    phase_plan=[dataclasses.asdict(Phase("realsemi", 1, 1))],
    optimizer_step=1,
)
args = SimpleNamespace(
    algorithm="realsemi",
    tp_size=1,
    shard_draft_by_tp=False,
    disaggregate=False,
    accumulation_steps=1,
    batch_size=1,
    target_model_path="/data/temp/artifacts/models/Qwen3-4B",
)
path = save_checkpoint(
    output_dir=str(Path(__file__).parent / "checkpoint_integration"),
    name="saved",
    fsdp_model=fsdp,
    draft_model=model,
    optimizer=optimizer,
    metadata=metadata,
    training_args=args,
)
state = load_distributed_training_state(path)
assert state["realsemi_config"]["serial_context_mode"] == "last"
assert optimizer.load_state_dict(state)
with FSDP.summon_full_params(fsdp):
    reloaded = RealSemiDraftModel.from_pretrained(path, attn_implementation="sdpa")
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(
            tensor.cpu(), reloaded.state_dict()[name], rtol=0, atol=0, check_dtype=False
        )
dist.barrier()
if rank == 0:
    print("CHECKPOINT_FSDP_MUON_RESUME_PASSED", flush=True)
dist.destroy_process_group()
