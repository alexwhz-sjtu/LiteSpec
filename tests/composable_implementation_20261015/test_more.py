import copy
import torch
from torch import nn
import pytest
from test_implementation import config, fixture
from litespec.core.composable.training_model import ComposableTrainingModel
from litespec.core.composable.batch import (
    prepare_layout,
    cpu_copy,
    PreparedLogicalBatch,
)
from litespec.modeling.draft.composable import ModelInputs, SharedOps
from litespec.modeling.draft.composable.contracts import ContextOutput
from litespec.modeling.draft.composable.backbone.attention.access.base import (
    build_visibility,
)
from litespec.modeling.draft.dflash import Qwen3DFlashAttention, Qwen3DFlashMLAAttention
from litespec.modeling.draft.dflash_kernels import DEFAULT_DFLASH_KERNELS
from litespec.training.optimizer import register_optimizer_parameters


@pytest.mark.parametrize("representation", ["mha", "mla"])
def test_attention_matches_existing_projection_and_math(representation):
    cfg = config(heterogeneous=representation == "mla")
    model, *_ = fixture(cfg)
    layer = model.backbone.layers[1 if representation == "mla" else 0]
    attention = layer.self_attn
    old_cls = (
        Qwen3DFlashMLAAttention if representation == "mla" else Qwen3DFlashAttention
    )
    old_cfg = copy.deepcopy(attention.config)
    old_cfg._attn_implementation = "eager"
    original = old_cls(old_cfg, attention.layer_idx, DEFAULT_DFLASH_KERNELS)
    original.load_state_dict(attention.state_dict())
    hidden = torch.randn(2, 4, 32, requires_grad=True)
    target = torch.randn(2, 5, 32)
    positions = torch.arange(9)[None].expand(2, -1)
    embeddings = layer.rotary_emb(hidden, positions)
    inputs = ModelInputs(
        target, hidden, torch.full((2, 1), 5), torch.ones(2, 1, dtype=torch.bool)
    )
    mask = build_visibility(
        inputs,
        positions[:, :5],
        torch.ones(2, 5, dtype=torch.bool),
        torch.zeros(2, 5, dtype=torch.long),
        4,
    )
    actual = attention(hidden, target, embeddings, inputs)
    expected = original(hidden, target, embeddings, mask)[0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), hidden, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), hidden)[0],
        atol=1e-6,
        rtol=1e-5,
    )


def test_conv_identity_start_and_no_cross_block():
    cfg = config("dflare", "rnn")
    cfg.composable_config["backbone"]["layers"][0]["local_operator"] = {
        "type": "dflash2_conv",
        "kernel_size": 2,
        "group_size": 8,
    }
    model, *_ = fixture(cfg)
    conv = model.backbone.layers[0].attention_conv
    hidden = torch.randn(2, 8, 32)
    prepared, coefficients = conv.prepare(hidden)
    torch.testing.assert_close(prepared, hidden)
    torch.testing.assert_close(conv.finish(prepared, coefficients), hidden)
    with torch.no_grad():
        conv.base_kernel[:, 1].fill_(1)
    impulse = torch.zeros(1, 8, 32)
    impulse[:, 3] = 1
    prepared, _ = conv.prepare(impulse)
    assert prepared[:, 4].eq(0).all()
    register_optimizer_parameters(
        model
    )  # DFlare weights, conv base and recursive heads must be classified.


def test_dflare_prior_and_fusion_formula():
    model, *_ = fixture(config("dflare"))
    fusion = model.context
    torch.testing.assert_close(
        fusion.layer_fusion_weights, torch.tensor([[0.0, -2.0], [-2.0, 0.0]])
    )
    target = torch.randn(2, 5, 2, 32)
    inputs = ModelInputs(
        target,
        torch.empty(2, 4, 32),
        torch.zeros(2, 1, dtype=torch.long),
        torch.ones(2, 1, dtype=torch.bool),
    )
    output = fusion(inputs)
    for index in range(2):
        weights = fusion.layer_fusion_weights[index].softmax(-1)
        expected = fusion.norm((target * weights[None, None, :, None]).sum(2))
        torch.testing.assert_close(output.context_hidden_states[index], expected)


def test_zero_supervision_and_fixed_anchor_width():
    ids = torch.zeros(2, 8, dtype=torch.long)
    mask = torch.zeros(2, 8)
    layout = prepare_layout(ids, mask, 4, 7)
    assert layout.anchor_positions.shape == (2, 7) and not layout.block_keep_mask.any()
    model, embedding, head, ids, hidden, teacher, mask = fixture(config(objective="bv"))
    wrapper = ComposableTrainingModel(model, head, embedding, 0, num_anchors=2)
    out = wrapper(ids, hidden, mask * 0, teacher)
    assert out.loss_terms.numerator == 0 and out.loss_terms.denominator == 0


def test_context_padding_and_document_isolation():
    target = torch.empty(1, 6, 1)
    inputs = ModelInputs(
        target,
        torch.empty(1, 4, 1),
        torch.tensor([[5]]),
        torch.ones(1, 1, dtype=torch.bool),
        anchor_document_ids=torch.tensor([[1]]),
    )
    mask = build_visibility(
        inputs,
        torch.arange(6)[None],
        torch.tensor([[0, 1, 1, 1, 1, 1]], dtype=torch.bool),
        torch.tensor([[0, 0, 0, 1, 1, 1]]),
        4,
    )[0, 0]
    assert mask[:, :6].nonzero()[:, 1].unique().tolist() == [3, 4]


def test_ragged_cache_physically_bounded():
    from litespec.modeling.draft.composable import DraftState

    model, embedding, head, ids, hidden, teacher, mask = fixture(config(window=True))
    model.eval()
    draft_state = DraftState()
    inputs = ModelInputs(
        hidden[:, :5],
        embedding(torch.tensor([[1, 0, 0, 0]]).expand(2, -1)),
        torch.tensor([[5], [3]]),
        torch.ones(2, 1, dtype=torch.bool),
        context_valid_mask=torch.tensor(
            [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool
        ),
        draft_state=draft_state,
        mode="features",
    )
    with torch.no_grad():
        model(inputs)
    for cache in draft_state.layers.values():
        assert cache.key_states.size(2) == 3
        assert cache.valid.sum(-1).eq(3).all()
        assert cache.positions.tolist() == [[2, 3, 4], [0, 1, 2]]


def test_export_forward_equivalence(tmp_path):
    import os, sys, subprocess
    from litespec.training.checkpoint import export_draft

    model, embedding, head, ids, hidden, teacher, mask = fixture(
        config("dflare", "rnn", heterogeneous=True, window=True)
    )
    model.eval()
    with torch.no_grad():
        model.proposal.markov_head.markov_w2.weight.normal_(std=0.02)
    model_inputs = ModelInputs(
        hidden[:1, :5],
        embedding(torch.tensor([[2, 0, 0, 0]])),
        torch.tensor([[5]]),
        torch.ones(1, 1, dtype=torch.bool),
        previous_token_ids=torch.tensor([[[2, 3, 4]]]),
    )
    with torch.no_grad():
        expected = model(
            model_inputs, shared_ops=SharedOps(embedding, head)
        ).draft_logits
    payload = {"inputs": vars(model_inputs), "head": head.weight, "expected": expected}
    torch.save(payload, tmp_path / "fixture.pt")
    export_draft(model, tmp_path)
    code = """from transformers import AutoModel
from types import SimpleNamespace
import torch,sys
p=sys.argv[1];payload=torch.load(p+'/fixture.pt',weights_only=True)
m=AutoModel.from_pretrained(p,trust_remote_code=True).eval()
head=torch.nn.Linear(32,23,bias=False);head.weight.data.copy_(payload['head'])
with torch.no_grad():
 result=m(SimpleNamespace(**payload['inputs']),shared_ops=SimpleNamespace(lm_head=head)).draft_logits
torch.testing.assert_close(result,payload['expected'])
assert not any(k.startswith('litespec') for k in sys.modules)
print('COLD_FORWARD_EQUIVALENT')
"""
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


def test_weighted_ce_fractional_denominator():
    from types import SimpleNamespace
    from litespec.core.composable.objectives import build_objective
    from litespec.core.composable.objectives.base import Supervision

    logits = torch.randn(1, 1, 2, 7, requires_grad=True)
    labels = torch.tensor([[[1, 2]]])
    mask = torch.tensor([[[0.0, 1.0]]])
    output = build_objective({"type": "hard_ce", "loss_decay_gamma": 0.1})(
        SimpleNamespace(draft_logits=logits), Supervision(labels, mask)
    )
    expected = torch.nn.functional.cross_entropy(
        logits[..., 1, :].flatten(0, 1), labels[..., 1].flatten()
    )
    torch.testing.assert_close(output.loss, expected)


def test_batched_rollback_limits():
    from litespec.modeling.draft.composable.contracts import DraftState, LayerState

    positions = torch.tensor([[0, 1, 2], [4, 5, 6]])
    state = DraftState(
        {
            0: LayerState(
                torch.randn(2, 2, 3, 4),
                torch.randn(2, 2, 3, 4),
                positions,
                torch.ones(2, 3, dtype=torch.bool),
                torch.zeros_like(positions),
            )
        }
    )
    state.rollback(torch.tensor([2, 5]))
    assert state.layers[0].valid.tolist() == [[True, True], [True, False]]
