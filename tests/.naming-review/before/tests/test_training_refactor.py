"""Behavioral checks for shared training and independent inference exports."""

import json
import os
import subprocess
import sys

import pytest
import torch
from refactor_baseline import capture

from litespec.core.block_metrics import block_accepted_length_terms
from litespec.core.outputs import RatioTerms, TrainingOutput
from litespec.distributed.transport import PacketReceiver
from litespec.training.checkpoint import export_draft
from litespec.training.optimizer import BF16Optimizer
from litespec.training.runner import OptimizerWindow


def _distributed_window_worker(rank, rendezvous):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        parameter = torch.nn.Parameter(torch.tensor(0.25))

        class RecordingOptimizer:
            def scale_model_gradients(self, factor):
                parameter.grad.mul_(factor)

            def step(self):
                # Simulate FSDP averaging below; production normalization and
                # metric collectives execute on two real CPU ranks.
                assert parameter.grad.item() == pytest.approx(5.0)
                return parameter.grad.abs().item()

        window = OptimizerWindow(RecordingOptimizer(), accumulation_steps=3)
        for x in (rank + 1, rank + 3):
            weight = float(x)
            numerator = (parameter * x).square() * weight
            window.backward(
                TrainingOutput(
                    numerator / weight, RatioTerms(numerator, torch.tensor(weight))
                )
            )
        dist.all_reduce(parameter.grad)
        parameter.grad.div_(2)
        window.step()
        assert window.metrics.reduce()["objective/loss"].item() == pytest.approx(0.625)
    finally:
        dist.destroy_process_group()


def test_two_rank_window_matches_global_large_batch(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_window_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2
    )


def test_acceptance_counts_all_overlapping_blocks():
    labels = torch.ones(1, 3, 3, dtype=torch.long)
    predictions = torch.tensor([[[1, 1, 1], [0, 1, 1], [1, 0, 1]]])
    terms = block_accepted_length_terms(
        predictions, labels, torch.ones_like(labels, dtype=torch.bool)
    )
    assert (terms.numerator / terms.denominator).item() == pytest.approx(7 / 3)
    empty = block_accepted_length_terms(
        predictions, labels, torch.zeros_like(labels, dtype=torch.bool)
    )
    assert empty.numerator == 0 and empty.denominator == 0


def test_optimizer_window_matches_large_batch_with_tail():
    def make():
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(0.2)
        return model, BF16Optimizer(model, lr=0.1, max_grad_norm=100, total_steps=5)

    actual, optimizer = make()
    expected, reference = make()
    window = OptimizerWindow(optimizer, accumulation_steps=4)
    total = torch.tensor(0.0)
    for x, weight in ((1.0, 1.0), (2.0, 3.0), (3.0, 2.0)):
        value = torch.tensor([[x]])
        numerator = actual(value).square().sum() * weight
        total = total + expected(value).square().sum() * weight
        window.backward(
            TrainingOutput(
                numerator / weight, RatioTerms(numerator, torch.tensor(weight))
            )
        )
    (total / 6).backward()
    reference.step()
    window.step()
    torch.testing.assert_close(actual.weight, expected.weight)
    assert window.micro_steps == 0
    torch.testing.assert_close(
        window.metrics.reduce()["objective/loss"], total.detach() / 6
    )


def test_optimizer_resume_allows_frozen_state_but_rejects_wrong_order():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    optimizer = BF16Optimizer(model, lr=0.1, parameters=list(model.parameters()))
    model[1].requires_grad_(False)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    state = optimizer.state_dict()
    restored = BF16Optimizer(model, lr=0.1, parameters=list(model.parameters()))
    assert restored.load_state_dict(state)
    invalid = {
        **state,
        "optimizer_parameter_names": list(reversed(state["optimizer_parameter_names"])),
    }
    with pytest.raises(ValueError, match="names/order"):
        restored.load_state_dict(invalid)


def test_two_stage_optimizer_order_and_scheduler_resume():
    import copy

    from litespec.training.recipes.dlite import (
        _set_student_stage1_trainable,
        _set_student_stage2_trainable,
        two_stage_optimizer_parameters,
    )

    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Deliberately register the head first: stage ordering must win.
            self.sequential_head = torch.nn.Linear(2, 1)
            self.backbone = torch.nn.Linear(2, 2)

        def forward(self, value):
            return self.sequential_head(self.backbone(value))

    model = Student()
    parameters = two_stage_optimizer_parameters(model)
    names = {id(p): name for name, p in model.named_parameters()}
    assert [names[id(p)] for p in parameters] == [
        "backbone.weight",
        "backbone.bias",
        "sequential_head.weight",
        "sequential_head.bias",
    ]
    optimizer = BF16Optimizer(
        model, parameters=parameters, lr=0.01, total_steps=10, warmup_ratio=0.2
    )
    _set_student_stage1_trainable(model)
    original_head = copy.deepcopy(model.sequential_head.state_dict())
    for _ in range(2):
        model(torch.ones(1, 2)).square().sum().backward()
        optimizer.step()
    for name, value in model.sequential_head.state_dict().items():
        torch.testing.assert_close(value, original_head[name], rtol=0, atol=0)
    checkpoint = copy.deepcopy(optimizer.state_dict())
    assert len(checkpoint["optimizer_state_dict"]["state"]) == 2
    restored_model = copy.deepcopy(model)
    restored = BF16Optimizer(
        restored_model,
        parameters=two_stage_optimizer_parameters(restored_model),
        lr=0.01,
        total_steps=10,
        warmup_ratio=0.2,
    )
    restored.load_state_dict(checkpoint)
    _set_student_stage2_trainable(model)
    _set_student_stage2_trainable(restored_model)
    for _ in range(3):
        for student, opt in ((model, optimizer), (restored_model, restored)):
            student(torch.ones(1, 2)).square().sum().backward()
            opt.step()
        assert optimizer.get_learning_rate() == restored.get_learning_rate()
        for actual, expected in zip(
            model.parameters(), restored_model.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    bad_shape = copy.deepcopy(checkpoint)
    bad_shape["optimizer_state_dict"]["state"][0]["exp_avg"] = torch.zeros(4)
    with pytest.raises(ValueError, match="shapes"):
        restored.load_state_dict(bad_shape)


@pytest.mark.parametrize(
    "field,value",
    [
        ("algorithm", "dspark"),
        ("mode", "teacher"),
        ("tp_size", 2),
        ("shard_draft_by_tp", True),
    ],
)
def test_resume_requires_current_algorithm_identity(field, value):
    from types import SimpleNamespace

    from litespec.training.checkpoint import (
        training_identity,
        validate_training_identity,
    )

    args = SimpleNamespace(
        algorithm="dlite",
        mode="sft",
        tp_size=1,
        shard_draft_by_tp=False,
        disaggregate=False,
        accumulation_steps=2,
        batch_size=1,
        target_model_path="target",
    )
    state = training_identity(args)
    validate_training_identity(state, args)
    with pytest.raises(ValueError, match=field):
        validate_training_identity({**state, field: value}, args)
    with pytest.raises(ValueError, match="current LiteSpec"):
        validate_training_identity({}, args)


def test_dlite_config_builder_replaces_target_architecture(monkeypatch):
    from types import SimpleNamespace

    from transformers import Qwen3Config

    from litespec.modeling.draft.dlite import DLiteDraftModel
    from litespec.training.recipes import dlite as common

    target = Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
    )
    monkeypatch.setattr(
        common, "load_text_model_config", lambda *args, **kwargs: target
    )
    args = SimpleNamespace(
        target_model_path="target",
        trust_remote_code=False,
        num_draft_layers=1,
        block_size=4,
        target_layer_ids="1",
        sequential_head="rnn",
        sequential_rank=4,
        mask_token_id=31,
        attention_backend="eager",
    )
    config = common.build_draft_config(args, model_role="student")
    assert config.architectures == ["DLiteDraftModel"]
    assert DLiteDraftModel(config).block_size == 4


def test_packet_ring_reuses_only_consumed_slots():
    class Transport:
        def receive(self, packet, *, batch_id, **kwargs):
            packet["id"] = batch_id
            return batch_id

        def wait_receive(self, handle):
            return 0.5

    receiver = PacketReceiver(
        Transport(), [{}, {}, {}], start=2, stop=10, phase_id=1, schema_id=2
    )
    assert [(index, packet["id"]) for index, packet in receiver] == [
        (i, i) for i in range(2, 10)
    ]
    assert receiver.wait_ms == 4.0


@pytest.fixture(scope="module")
def tiny_models():
    return capture(models_only=True)


@pytest.mark.parametrize("algorithm", ["dflash", "dflash2", "dspark"])
def test_benchmark_uses_independent_model_config(monkeypatch, tiny_models, algorithm):
    from types import SimpleNamespace

    from evaluation import dflash_model_loading as loading

    draft = tiny_models[algorithm]
    monkeypatch.setattr(
        loading.AutoConfig, "from_pretrained", lambda *a, **k: draft.config
    )
    monkeypatch.setattr(type(draft), "from_pretrained", lambda *a, **k: draft)
    monkeypatch.setattr(
        loading.AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **k: torch.nn.Linear(2, 2),
    )
    monkeypatch.setattr(
        loading.AutoTokenizer,
        "from_pretrained",
        lambda *a, **k: SimpleNamespace(mask_token_id=31),
    )
    args = SimpleNamespace(
        algorithm=algorithm,
        draft_name_or_path="draft",
        model_name_or_path="target",
        trust_remote_code=False,
        mask_token_id=30,
    )
    _, loaded, _ = loading.load_dflash_benchmark_models(args, "cpu")
    assert loaded is draft
    assert getattr(loaded.config, f"{algorithm}_config")["mask_token_id"] == 30
    args.algorithm = "dlite"
    with pytest.raises(ValueError, match="cannot load"):
        loading.load_dflash_benchmark_models(args, "cpu")


@pytest.mark.parametrize("algorithm", ["dflash", "dflash2", "dspark"])
def test_training_wrappers_run_and_return_block_metrics(tiny_models, algorithm):
    import importlib

    model = tiny_models[algorithm]
    model.zero_grad(set_to_none=True)
    cls = getattr(
        importlib.import_module(f"litespec.core.{algorithm}"),
        {
            "dflash": "OnlineDFlashModel",
            "dflash2": "OnlineDFlash2Model",
            "dspark": "OnlineDSparkModel",
        }[algorithm],
    )
    wrapper = cls(
        model,
        torch.nn.Linear(16, 32, bias=False),
        torch.nn.Embedding(32, 16),
        mask_token_id=31,
        block_size=4,
        num_anchors=3,
        attention_backend="eager",
        objective_chunk_blocks=2,
    )
    ids = torch.arange(12).view(1, -1)
    output = wrapper(
        ids,
        torch.randn(1, 12, 16),
        torch.ones_like(ids),
        target_last_hidden_states=torch.randn(1, 12, 16),
    )
    for key in ("hard_label/block_accepted_length", "expected_accept_len"):
        terms = output.ratio_metrics[key]
        value = terms.numerator / terms.denominator
        assert 1 <= value <= 4
    output.loss_terms.numerator.backward()
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize(
    "algorithm,mode",
    [
        ("dflash", None),
        ("dflash2", None),
        ("dspark", None),
        ("dlite", "sft"),
        ("dlite", "teacher"),
        ("dlite", "two_stage"),
    ],
)
def test_all_training_parsers_accept_effective_cli(algorithm, mode):
    from litespec.training.algorithms import get_training_adapter

    arguments = [
        "--target-model-path",
        "target",
        "--train-data-path",
        "train.jsonl",
        "--output-dir",
        "out",
        "--batch-size",
        "3",
        "--batch-size",
        "2",
        "--profile",
        "--no-profile",
    ]
    if mode == "two_stage":
        arguments += [
            "--teacher-draft-path",
            "teacher",
            "--stage1-epochs",
            "1",
            "--stage2-epochs",
            "1",
            "--learning-rate",
            "0.0004",
            "--warmup-ratio",
            "0.04",
        ]
    args = get_training_adapter(algorithm).parse_args(mode, argv=arguments)
    assert args.batch_size == 2 and not args.profile
    assert hasattr(args, "mode") == (algorithm == "dlite")


@pytest.mark.parametrize("algorithm", ["dflash", "dflash2", "dspark", "dlite"])
def test_export_runs_without_installed_training_package(
    tmp_path, tiny_models, algorithm
):
    model = tiny_models[algorithm]
    exported = tmp_path / "model"
    export_draft(model, exported)
    config = json.loads((exported / "config.json").read_text())
    assert config["architectures"] == [type(model).__name__]
    keys = {"dflash_config", "dflash2_config", "dspark_config", "dlite_config"}
    assert keys.intersection(config) == {f"{algorithm}_config"}
    code = """
import importlib.abc
import sys
class RejectTrainingPackage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "litespec" or fullname.startswith("litespec."):
            raise ImportError("The exported model must not import litespec")
sys.meta_path.insert(0, RejectTrainingPackage())
import torch
from transformers import AutoModel
model = AutoModel.from_pretrained(sys.argv[1], trust_remote_code=True, attn_implementation="eager")
dlite = model.__class__.__name__ == "DLiteDraftModel"
noise = torch.randn(1, 4, 16)
target = torch.randn(1, 1, 1, 16) if dlite else torch.randn(1, 1, 16)
out = model(position_ids=torch.arange(4 if dlite else 5).view(1,-1),
            noise_embedding=noise, target_hidden=target,
            attention_mask=torch.zeros(1,1,4,5), rotary_position_ids=torch.arange(5).view(1,-1),
            **({"history_hidden": torch.empty(1,1,0,16)} if dlite else {}))
assert out.shape == (1,4,16)
assert torch.isfinite(out).all()
"""
    env = {
        **os.environ,
        "HF_MODULES_CACHE": str(tmp_path / "modules"),
        "HF_HUB_OFFLINE": "1",
    }
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(exported)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
