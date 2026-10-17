"""BV uses actual full-vocabulary target trajectories; also check Flex parity."""

import copy
import json
from pathlib import Path
import torch
from transformers import Qwen3ForCausalLM
from test_implementation import config
from litespec.modeling.draft.composable import ComposableDraftModel
from litespec.core.composable.training_model import ComposableTrainingModel
from litespec.core.composable.batch import prepare_layout

root = Path(__file__).resolve().parent
torch.manual_seed(104)
target = (
    Qwen3ForCausalLM.from_pretrained(
        root / "fixtures/target", torch_dtype=torch.bfloat16
    )
    .cuda()
    .eval()
    .requires_grad_(False)
)
with torch.no_grad():
    ids = target.generate(
        torch.tensor([[6, 9]], device="cuda"),
        attention_mask=torch.ones(1, 2, device="cuda"),
        max_new_tokens=18,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        pad_token_id=0,
        eos_token_id=None,
    )
    outputs = target.model(ids, output_hidden_states=True, use_cache=False)
    hidden = torch.cat([outputs.hidden_states[i + 1] for i in (1, 4)], -1)
    teacher = outputs.hidden_states[-1]
mask = torch.ones_like(ids, dtype=torch.float32)
mask[:, :2] = 0
records = []
for proposal in ("parallel", "markov", "rnn", "sequential_rnn"):
    cfg = config("dflare", proposal, "bv", heterogeneous=True, window=True)
    cfg.vocab_size = 128
    model = ComposableDraftModel(cfg).cuda().bfloat16()
    wrapper = ComposableTrainingModel(
        model,
        target.lm_head,
        target.model.embed_tokens,
        0,
        num_anchors=2,
        objective_chunk_blocks=1,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for epoch in (0.0, 1.5, 3.0):
        optimizer.zero_grad()
        output = wrapper(ids, hidden, mask, teacher, epoch_progress=epoch)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        optimizer.step()
        records.append(
            {
                "proposal": proposal,
                "epoch_progress": epoch,
                "loss": output.loss.item(),
                "annealing_floor": wrapper.objective.annealing_score_floor,
            }
        )
# Reuse exact weights and anchor layout for eager/SDPA/Flex forward and gradients.
cfg = config(heterogeneous=True, window=True)
cfg.vocab_size = 128
cfg.head_dim = 16
cfg.composable_config["backbone"]["layers"][1]["attention"].update(
    qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=16
)
base = ComposableDraftModel(cfg).cuda()
other_cfg = copy.deepcopy(cfg)
for layer in other_cfg.composable_config["backbone"]["layers"]:
    layer["attention"]["backend"] = "flex_attention"
flex = ComposableDraftModel(other_cfg).cuda()
flex.load_state_dict(base.state_dict())
embedding = torch.nn.Embedding(128, 32, device="cuda").requires_grad_(False)
head = torch.nn.Linear(32, 128, bias=False, device="cuda").requires_grad_(False)
layout = prepare_layout(ids, mask, 4, 2)
values = []
for model in (base, flex):
    wrapper = ComposableTrainingModel(
        model, head, embedding, 0, num_anchors=2, objective_chunk_blocks=1
    )
    out = wrapper(ids, hidden.float(), mask, teacher.float(), layout=layout)
    values.append(out.loss.detach())
    out.loss.backward()
torch.testing.assert_close(values[0], values[1], atol=2e-5, rtol=2e-5)
for (name, p), (_, q) in zip(base.named_parameters(), flex.named_parameters()):
    torch.testing.assert_close(p.grad, q.grad, atol=2e-4, rtol=2e-3, msg=name)
(root / "bv_flex_gpu_result.json").write_text(
    json.dumps({"bv": records, "flex_loss": values[1].item()}, indent=2)
)
print("BV_TRAJECTORY_AND_FLEX_GPU_OK", flush=True)
