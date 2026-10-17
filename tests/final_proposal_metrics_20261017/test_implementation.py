"""Method and integration checks; all fixtures/exports stay under tests."""

import ast
import copy
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from litespec.modeling.draft.composable import (
    ComposableDraftConfig,
    ComposableDraftModel,
    ModelInputs,
    DraftState,
    SharedOps,
)
from litespec.modeling.draft.composable.backbone.attention.access.base import (
    build_visibility,
)
from litespec.core.composable.batch import prepare_layout
from litespec.core.composable.training_model import ComposableTrainingModel
from litespec.core.composable.objectives import build_objective
from litespec.core.composable.objectives.base import Supervision
from litespec.core.composable.objectives.bv import integrated_log_scores
from litespec.core.outputs import RatioTerms

ROOT = Path(__file__).resolve().parents[2]


def config(
    context="dflash",
    proposal="parallel",
    objective="hard_ce",
    heterogeneous=False,
    window=False,
):
    layers = [
        {"attention": {"representation": "mha", "access": "dense", "backend": "sdpa"}}
        for _ in range(2)
    ]
    if heterogeneous:
        layers[1]["attention"].update(
            representation="mla",
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=8,
        )
    if window:
        for layer in layers:
            layer["attention"].update(access="sliding_window", window_size=3)
    recipe = {
        "context": {"type": context, "target_layer_ids": [1, 5]},
        "backbone": {"type": "transformer", "layers": layers},
        "proposal": {"type": proposal, "rank": 8, "top_k": 5},
        "objective": {"type": objective, "loss_decay_gamma": 2.0},
    }
    return ComposableDraftConfig(
        hidden_size=32,
        intermediate_size=48,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        num_target_layers=8,
        vocab_size=23,
        block_size=4,
        composable_config=recipe,
        attention_dropout=0.0,
    )


def fixture(cfg=None, batch_size=2):
    cfg = cfg or config()
    model = ComposableDraftModel(cfg)
    embedding, head = (
        nn.Embedding(cfg.vocab_size, cfg.hidden_size),
        nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False),
    )
    embedding.requires_grad_(False)
    head.requires_grad_(False)
    ids = torch.randint(cfg.vocab_size, (batch_size, 10))
    hidden = torch.randn(batch_size, 10, 64)
    teacher = torch.randn(batch_size, 10, 32)
    mask = torch.ones(batch_size, 10)
    mask[-1, -2:] = 0
    return model, embedding, head, ids, hidden, teacher, mask


@pytest.mark.parametrize("context", ["dflash", "dflare"])
@pytest.mark.parametrize(
    "proposal", ["parallel", "markov", "rnn", "sequential_rnn", "selector"]
)
def test_train_and_sample(context, proposal):
    torch.manual_seed(7)
    model, embedding, head, ids, hidden, teacher, mask = fixture(
        config(context, proposal, heterogeneous=True, window=True)
    )
    wrapper = ComposableTrainingModel(
        model, head, embedding, 0, num_anchors=2, objective_chunk_blocks=1
    )
    out = wrapper(ids, hidden, mask, teacher)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    model.eval()
    noise = embedding(torch.tensor([[2, 0, 0, 0]]).expand(2, -1))
    inputs = ModelInputs(
        hidden[:, :4],
        noise,
        torch.full((2, 1), 4),
        torch.ones(2, 1, dtype=torch.bool),
        anchor_token_ids=torch.full((2, 1), 2),
        mode="sample",
        temperature=1.0,
    )
    result = model(inputs, shared_ops=SharedOps(embedding, head))
    assert result.draft_token_ids.shape == (2, 1, 3)
    torch.testing.assert_close(result.draft_probabilities.sum(-1), torch.ones(2, 1, 3))
    if proposal == "selector":
        assert (result.draft_probabilities > 0).sum(-1).eq(5).all()
    # Sampling rows must match teacher forcing along the sampled trajectory.
    inputs.mode = "teacher_forcing"
    inputs.previous_token_ids = torch.cat(
        (inputs.anchor_token_ids[..., None], result.draft_token_ids[..., :-1]), -1
    )
    scored = model(inputs, shared_ops=SharedOps(embedding, head))
    if proposal != "selector":
        torch.testing.assert_close(result.draft_logits, scored.draft_logits)
    else:
        sparse = torch.zeros_like(result.draft_probabilities).scatter(
            -1, scored.candidate_ids, scored.candidate_logits.float().softmax(-1)
        )
        torch.testing.assert_close(result.draft_probabilities, sparse)


@pytest.mark.parametrize(
    "objective", ["hard_ce", "soft_ce", "forward_kl", "tv", "nla", "dpace", "bv"]
)
def test_chunked_gradient_equals_full(objective):
    torch.manual_seed(9)
    cfg = config(objective=objective, proposal="rnn")
    model, embedding, head, ids, hidden, teacher, mask = fixture(cfg)
    other = copy.deepcopy(model)
    layout = prepare_layout(ids, mask, 4, 4, complete_blocks=objective == "bv")
    losses = []
    for candidate, chunk in [(model, 0), (other, 1)]:
        wrapper = ComposableTrainingModel(
            candidate, head, embedding, 0, num_anchors=4, objective_chunk_blocks=chunk
        )
        result = wrapper(ids, hidden, mask, teacher, layout=layout, epoch_progress=1.5)
        losses.append(result.loss.detach())
        result.loss.backward()
    torch.testing.assert_close(losses[0], losses[1])
    for (name, p), (_, q) in zip(model.named_parameters(), other.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5, msg=name)


def test_nla_frozen_source_loss_and_gradient():
    source = ROOT / "refrence/objectives/nla.py"
    tree = ast.parse(source.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "neg_log_acceptance_loss"
    )
    namespace = {"torch": torch, "_NLA_EPS": 1e-5}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    draft_logits = torch.randn(2, 3, 4, 7, requires_grad=True)
    teacher = torch.randn_like(draft_logits)
    mask = torch.randint(0, 2, (2, 3, 4)).float()
    weights = torch.exp(-torch.arange(4) / 2.0)
    element = namespace["neg_log_acceptance_loss"](draft_logits, teacher)
    expected = (
        (element * mask * weights).flatten(1).sum(-1) / (mask.flatten(1).sum(-1) + 1e-5)
    ).mean()
    actual = build_objective({"type": "nla", "loss_decay_gamma": 2.0})(
        SimpleNamespace(draft_logits=draft_logits),
        Supervision(torch.zeros_like(mask, dtype=torch.long), mask, teacher),
    ).loss
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, draft_logits, retain_graph=True)[0],
        torch.autograd.grad(expected, draft_logits)[0],
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dpace_frozen_source_loss_and_gradient(dtype):
    source = ROOT / "refrence/objectives/dpace.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "OnlineDFlashModel"
    )
    function = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_dpace_weight"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    draft_logits = torch.randn(2, 3, 4, 7, requires_grad=True, dtype=dtype)
    labels = torch.randint(7, (2, 3, 4))
    mask = torch.randint(0, 2, (2, 3, 4)).float()
    ce = torch.nn.functional.cross_entropy(
        draft_logits.flatten(0, -2), labels.flatten(), reduction="none"
    ).reshape_as(mask)
    with torch.no_grad():
        credit = namespace["_dpace_weight"](
            SimpleNamespace(dpace_alpha=0.5), ce.neg().exp(), mask, mask > 0, "dpace"
        )
    expected = (ce * mask * credit).sum() / 2
    actual = build_objective({"type": "dpace"})(
        SimpleNamespace(draft_logits=draft_logits), Supervision(labels, mask)
    ).loss
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, draft_logits, retain_graph=True)[0],
        torch.autograd.grad(expected, draft_logits)[0],
    )


def test_bv_integrated_formula_and_prefix_gradient():
    q = torch.softmax(torch.randn(1, 2, 3, 5), -1).requires_grad_()
    p = torch.softmax(torch.randn_like(q), -1)
    ids = torch.randint(5, (1, 2, 3))
    r = torch.ones(1, 2)
    a = torch.ones_like(r)
    scores = []
    for i in range(3):
        scores.append(
            torch.minimum(a[..., None] * p[..., i, :], r[..., None] * q[..., i, :]).sum(
                -1
            )
        )
        r = (
            r
            * q[..., i, :].gather(-1, ids[..., i, None]).squeeze(-1)
            / p[..., i, :].gather(-1, ids[..., i, None]).squeeze(-1)
        )
        a = torch.minimum(a, r)
    expected = torch.stack(scores, -1).log()
    actual = integrated_log_scores(q.log(), p.log(), ids)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), q, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), q)[0],
        atol=1e-5,
        rtol=1e-5,
    )


def test_bv_floor_follows_proposal():
    parallel = build_objective({"type": "bv"}, "parallel")
    for kind in ("markov", "rnn", "sequential_rnn"):
        assert build_objective({"type": "bv"}, kind).annealing_score_floor is None
    assert parallel.annealing_score_floor == 1e-6
    draft_logits = torch.tensor(
        [[[[0.0, -60.0], [0.0, -60.0], [0.0, -60.0]]]], requires_grad=True
    )
    teacher = -draft_logits.detach()
    supervision = Supervision(
        torch.ones(1, 1, 3, dtype=torch.long),
        torch.ones(1, 1, 3),
        teacher,
        epoch_progress=0,
    )
    plain = parallel(SimpleNamespace(draft_logits=draft_logits), supervision).loss
    corrected = build_objective({"type": "bv"}, "rnn")(
        SimpleNamespace(draft_logits=draft_logits), supervision
    ).loss
    assert plain < corrected
    torch.testing.assert_close(plain, torch.tensor(-math.log(1e-6)))
    with pytest.raises(ValueError):
        config(proposal="selector", objective="bv")


def test_swa_history_plus_whole_block():
    inputs = ModelInputs(
        torch.empty(1, 8, 1),
        torch.empty(1, 8, 1),
        torch.tensor([[3, 6]]),
        torch.ones(1, 2, dtype=torch.bool),
    )
    positions = torch.arange(8)[None]
    valid = torch.ones(1, 8, dtype=torch.bool)
    mask = build_visibility(
        inputs, positions, valid, torch.zeros_like(positions), 8, 2
    )[0, 0]
    assert mask[0, :8].nonzero().flatten().tolist() == [1, 2]
    assert mask[3, :8].nonzero().flatten().tolist() == [1, 2]
    assert mask[4, :8].nonzero().flatten().tolist() == [4, 5]
    assert mask[:4, 8:12].all() and not mask[:4, 12:].any()
    assert mask[4:, 12:].all() and not mask[4:, 8:12].any()


@pytest.mark.parametrize("heterogeneous", [False, True])
def test_confirmed_cache_and_rollback(heterogeneous):
    torch.manual_seed(11)
    model, embedding, head, ids, hidden, teacher, mask = fixture(
        config(heterogeneous=heterogeneous, window=True), batch_size=1
    )
    model.eval()
    ops = SharedOps(embedding, head)
    draft_state = DraftState()

    def inputs(start, end, draft_state=None):
        return ModelInputs(
            hidden[:, start:end],
            embedding(torch.tensor([[1, 0, 0, 0]])),
            torch.tensor([[end]]),
            torch.ones(1, 1, dtype=torch.bool),
            context_position_ids=torch.arange(start, end)[None],
            anchor_token_ids=torch.tensor([[1]]),
            mode="sample",
            draft_state=draft_state,
        )

    with torch.no_grad():
        model(inputs(0, 2, draft_state), shared_ops=ops)
        cached = model(inputs(2, 5, draft_state), shared_ops=ops)
        full = model(inputs(0, 5), shared_ops=ops)
        torch.testing.assert_close(cached.draft_logits, full.draft_logits)
        assert all(
            s.positions.min() >= 2 and s.positions.max() < 5
            for s in draft_state.layers.values()
        )
        draft_state.rollback(4)
        rolled = model(inputs(4, 5, draft_state), shared_ops=ops)
        torch.testing.assert_close(rolled.draft_logits, full.draft_logits)


@pytest.mark.parametrize("objective", ["hard_ce", "nla", "dpace"])
def test_selector_native_mixed_gradients(objective):
    torch.manual_seed(17)
    model, embedding, head, ids, hidden, teacher, mask = fixture(
        config(proposal="selector", objective=objective), batch_size=3
    )
    reference = copy.deepcopy(model)
    layout = prepare_layout(ids, mask, 4, 3)
    wrapper = ComposableTrainingModel(
        model, head, embedding, 0, objective_chunk_blocks=1
    )
    complete = wrapper(
        ids, hidden, mask, teacher, layout=layout, selector_loss_alpha=0.7
    )
    complete.loss.backward()
    main_den = (
        torch.tensor(3.0)
        if objective in ("nla", "dpace")
        else __import__("litespec.core.composable.batch", fromlist=["position_weights"])
        .position_weights(layout.prediction_mask, 2.0)
        .sum()
    )
    totals = torch.tensor(
        [
            3.0,
            main_den.item(),
            __import__("litespec.core.composable.batch", fromlist=["position_weights"])
            .position_weights(layout.prediction_mask, 2.0)
            .sum()
            .item(),
        ]
    )
    split = ComposableTrainingModel(
        reference, head, embedding, 0, objective_chunk_blocks=1
    )
    for start, end in [(0, 1), (1, 3)]:
        out = split(
            ids[start:end],
            hidden[start:end],
            mask[start:end],
            teacher[start:end],
            layout=layout.batch_slice(start, end),
            window_denominators=totals,
            selector_loss_alpha=0.7,
        )
        (out.loss_terms.numerator / 3).backward()
    for (name, p), (_, q) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-5, msg=name)


def test_export_cold_reload(tmp_path):
    from litespec.training.checkpoint import export_draft

    model, *_ = fixture(
        config("dflare", "rnn", "hard_ce", heterogeneous=True, window=True)
    )
    export_draft(model, tmp_path)
    code = """from transformers import AutoConfig,AutoModel
import torch
p=__import__('sys').argv[1]
c=AutoConfig.from_pretrained(p,trust_remote_code=True)
m=AutoModel.from_pretrained(p,trust_remote_code=True)
assert type(m).__name__=='ComposableDraftModel'
assert c.composable_config['backbone']['layers'][1]['attention']['representation']=='mla'
assert not any(k.startswith('litespec') for k in __import__('sys').modules)
print('COLD_RELOAD_OK')
"""
    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_config_inherits_explicit_head_dim():
    c = config()
    assert c.hidden_size == 32 and c.head_dim == 8
    c = ComposableDraftConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
    )
    assert c.head_dim == 16
    ComposableDraftModel(c)
