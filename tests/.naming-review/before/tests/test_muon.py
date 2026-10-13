"""Numerical reference and parameter ownership tests for hybrid Muon."""

import argparse
import copy

import pytest
import torch

from litespec.training.config import add_common_args
from litespec.training.fsdp_layout import ParameterSpec, local_layouts
from litespec.training.muon import (
    ShardedMuon,
    muon_config_from_args,
    validate_muon_config,
)
from litespec.training.optimizer import BF16Optimizer, register_optimizer_parameters


@pytest.mark.parametrize("shape", [(8, 8), (3, 11), (11, 3)])
@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("rule", ["original", "match_rms_adamw"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_matches_torch_muon(shape, nesterov, rule, device):
    torch.manual_seed(17)
    params = [torch.nn.Parameter(torch.randn(*shape, device=device)) for _ in range(3)]
    reference = [torch.nn.Parameter(p.detach().clone()) for p in params]
    specs = [ParameterSpec(str(i), shape, "muon") for i in range(3)]
    kwargs = dict(
        lr=0.02, momentum=0.95, weight_decay=0.1, nesterov=nesterov, adjust_lr_fn=rule
    )
    opt = ShardedMuon(params, local_layouts(specs), matrices_per_rank=2, **kwargs)
    ref = torch.optim.Muon(reference, **kwargs)
    for step in range(5):
        for i, (p, q) in enumerate(zip(params, reference)):
            grad = torch.zeros_like(p) if step == 2 else torch.randn_like(p)
            p.grad = None if step == 3 and i == 0 else grad
            q.grad = None if p.grad is None else grad.clone()
        opt.step()
        ref.step()
        for p, q in zip(params, reference):
            torch.testing.assert_close(p, q, rtol=0, atol=5e-4)
            torch.testing.assert_close(
                opt.state[p]["momentum_buffer"],
                ref.state[q]["momentum_buffer"],
                rtol=0,
                atol=0,
            )


class ToyHybrid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(7, 5)
        self.norm = torch.nn.LayerNorm(5)
        self.embedding = torch.nn.Embedding(9, 5)
        self.history_fuse = torch.nn.Linear(15, 5, bias=False)
        self.sequential_head = torch.nn.Linear(5, 5)
        self.markov_head = torch.nn.Linear(5, 5)
        self.confidence_head = torch.nn.Linear(5, 1)
        self.candidate_selector = torch.nn.Linear(5, 5)


def test_ownership_exceptions_and_unknown_parameters():
    model = ToyHybrid()
    parameters, specs = register_optimizer_parameters(model)
    assert len(parameters) == len(list(model.parameters()))
    assert {s.name for s in specs if s.optimizer == "muon"} == {
        "fc.weight",
        "history_fuse.weight",
    }
    model.register_parameter("unknown", torch.nn.Parameter(torch.ones(3, 3)))
    with pytest.raises(ValueError, match="Unclassified"):
        register_optimizer_parameters(model)


def test_tied_parameter_conflicting_ownership():
    model = torch.nn.Module()
    model.fc = torch.nn.Linear(5, 5, bias=False)
    model.embedding = torch.nn.Embedding(5, 5)
    model.embedding.weight = model.fc.weight
    with pytest.raises(ValueError, match="conflicting"):
        register_optimizer_parameters(model)


@pytest.mark.parametrize("algorithm", ["dflash", "dflash2", "dspark"])
def test_real_family_grouping(algorithm):
    from test_dflash_family import tiny_config

    from litespec.modeling.draft.dflash import DFlashDraftModel
    from litespec.modeling.draft.dflash2 import DFlash2DraftModel
    from litespec.modeling.draft.dspark import DSparkDraftModel

    cls = {
        "dflash": DFlashDraftModel,
        "dflash2": DFlash2DraftModel,
        "dspark": DSparkDraftModel,
    }[algorithm]
    model = cls(
        tiny_config(
            cls.__name__,
            conv_kernel_size=2,
            conv_group_size=4,
            selector_rank=4,
            selector_top_k=3,
            backbone_conv=True,
            markov_rank=4,
            markov_head_type="rnn",
            enable_confidence_head=True,
        )
    )
    _, specs = register_optimizer_parameters(model)
    groups = {s.name: s.optimizer for s in specs}
    assert groups["fc.weight"] == "muon"
    for name, kind in groups.items():
        if name.endswith("kernel_projection.weight"):
            assert kind == "muon"
        if any(
            part in name
            for part in (
                "base_kernel",
                "candidate_selector",
                "markov_head",
                "confidence_head",
            )
        ):
            assert kind == "adamw"


@pytest.mark.parametrize("teacher", [False, True])
def test_real_dlite_grouping(teacher):
    from refactor_baseline import capture

    from litespec.modeling.draft.dlite import DLiteDraftModel

    model = capture(models_only=True)["dlite"]
    if teacher:
        config = copy.deepcopy(model.config)
        config.dlite_config.update(
            model_role="swa_teacher", swa_window_size=4, history_layer_ids=[0, 2, 3]
        )
        model = DLiteDraftModel(config)
    _, specs = register_optimizer_parameters(model)
    groups = {s.name: s.optimizer for s in specs}
    assert groups["layer_depth_embedding.weight"] == "adamw"
    if teacher:
        assert groups["history_fuse.weight"] == "muon"
    assert all(
        kind == "adamw"
        for name, kind in groups.items()
        if name.startswith("sequential_head.")
    )


def test_hybrid_clip_freeze_and_resume():
    torch.manual_seed(13)
    model = ToyHybrid()
    opt = BF16Optimizer(
        model, lr=9, muon_config={}, total_steps=10, warmup_ratio=0.2, max_grad_norm=1
    )
    model.sequential_head.requires_grad_(False)
    initial_head = model.sequential_head.weight.detach().clone()
    for i in range(2):
        for p in model.parameters():
            p.grad = torch.full_like(p, 3) if p.requires_grad else None
        assert opt.step() > 1
    torch.testing.assert_close(
        model.sequential_head.weight, initial_head, rtol=0, atol=0
    )
    index = opt.parameter_names.index("sequential_head.weight")
    assert opt.fp32_params[index] not in opt.optimizer.state

    restored_model = copy.deepcopy(model)
    restored = BF16Optimizer(
        restored_model,
        lr=0.123,
        parameters=list(restored_model.parameters()),
        muon_config={},
        total_steps=10,
        warmup_ratio=0.2,
        max_grad_norm=1,
    )
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    for obj in (model, restored_model):
        obj.sequential_head.requires_grad_(True)
        obj.fc.requires_grad_(False)
    for _ in range(3):
        for a, b in zip(model.parameters(), restored_model.parameters()):
            a.grad = torch.randn_like(a) if a.requires_grad else None
            b.grad = a.grad.clone() if a.grad is not None else None
        opt.step()
        restored.step()
        for a, b in zip(opt.fp32_params, restored.fp32_params):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert opt.get_learning_rates() == restored.get_learning_rates()


def test_hybrid_clip_is_shared_with_adamw():
    model = torch.nn.Linear(5, 3)
    opt = BF16Optimizer(model, lr=0.1, muon_config={}, max_grad_norm=1, warmup_ratio=0)
    refs = [torch.nn.Parameter(p.detach().clone()) for p in model.parameters()]
    muon = torch.optim.Muon([refs[0]], lr=0.02, momentum=0.95, weight_decay=0.1)
    adamw = torch.optim.AdamW(
        [refs[1]], lr=0.0004, betas=(0.9, 0.999), weight_decay=0.1
    )
    for p, q in zip(model.parameters(), refs):
        p.grad = torch.full_like(p, 4)
        q.grad = p.grad.clone()
    expected_norm = torch.nn.utils.clip_grad_norm_(refs, 1)
    norm = opt.step()
    muon.step()
    adamw.step()
    assert norm == pytest.approx(expected_norm.item())
    for p, q in zip(opt.fp32_params, refs):
        torch.testing.assert_close(p, q, rtol=0, atol=5e-4)


def test_accumulation_tail_matches_large_batch():
    from litespec.core.outputs import RatioTerms, TrainingOutput
    from litespec.training.runner import OptimizerWindow

    torch.manual_seed(5)
    model = torch.nn.Linear(5, 3)
    reference = copy.deepcopy(model)
    actual = BF16Optimizer(model, lr=0.1, muon_config={}, warmup_ratio=0)
    expected = BF16Optimizer(reference, lr=0.1, muon_config={}, warmup_ratio=0)
    window = OptimizerWindow(actual, accumulation_steps=4)
    numerator, denominator = 0, 0
    for weight in (1.0, 3.0):
        inputs = torch.randn(2, 5)
        loss = model(inputs).square().sum() * weight
        window.backward(
            TrainingOutput(loss / weight, RatioTerms(loss, torch.tensor(weight)))
        )
        numerator = numerator + reference(inputs).square().sum() * weight
        denominator += weight
    (numerator / denominator).backward()
    expected.step()
    window.step()
    for a, b in zip(actual.fp32_params, expected.fp32_params):
        torch.testing.assert_close(a, b, rtol=0, atol=5e-4)


@pytest.mark.parametrize(
    "setting",
    [
        {"momentum": 1},
        {"lr": float("nan")},
        {"eps": 0},
        {"weight_decay": -1},
        {"ns_steps": 100},
        {"ns_coefficients": [1, 2]},
        {"adamw_betas": [0.9, 1]},
        {"matrices_per_rank": 0},
    ],
)
def test_invalid_config(setting):
    with pytest.raises(ValueError):
        validate_muon_config(setting)


def test_cli_and_disabled_settings():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    base = ["--target-model-path", "target", "--output-dir", "out"]
    assert muon_config_from_args(parser.parse_args(base + ["--muon-lr", "-1"])) is None
    args = parser.parse_args(base + ["--use-muon"])
    config = muon_config_from_args(args)
    assert config["lr"] == 0.02
    assert config["nesterov"] is True
    assert config["ns_steps"] == 5
    assert config["ns_coefficients"] == (3.4445, -4.775, 2.0315)
    assert config["eps"] == 1e-7
    assert config["adjust_lr_fn"] == "original"


def test_checkpoint_rejects_changed_layout_and_mode():
    model = ToyHybrid()
    opt = BF16Optimizer(model, lr=0.1, muon_config={})
    state = copy.deepcopy(opt.state_dict())
    state["optimizer_state_dict"]["layouts"][0]["shards"] = ((1, 35),)
    with pytest.raises(ValueError, match="shapes"):
        opt.load_state_dict(state)
    adam = BF16Optimizer(model, lr=0.1)
    with pytest.raises(ValueError, match="configuration"):
        opt.load_state_dict(adam.state_dict())
    with pytest.raises(ValueError, match="configuration"):
        adam.load_state_dict(opt.state_dict())
