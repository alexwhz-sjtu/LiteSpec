"""Exercise the actual lifecycle on CPU, without starting GPU training."""

from __future__ import annotations

import ast
import copy
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from litespec.core.outputs import RatioTerms, TrainingOutput
from litespec.training.algorithms import RunState, TrainingSession, get_training_adapter
from litespec.training.checkpoint import (
    capture_rng_state,
    load_distributed_training_state,
    restore_rng_state,
    save_distributed_training_state,
    session_metadata,
    training_identity,
    validate_session_resume,
    validate_training_identity,
)
from litespec.training.optimizer import BF16Optimizer
from litespec.training.muon import MUON_CLI_KEYS, MUON_DEFAULTS, muon_config_from_args
from litespec.training.recipes.dlite import _cosine_transition_scales
from litespec.training.runner import run_draft

WORKFLOWS = [
    ("dflash", None),
    ("dflash2", None),
    ("dspark", None),
    ("dlite", "sft"),
    ("dlite", "teacher"),
    ("dlite", "two_stage"),
]


class ToyDraft(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.backbone = torch.nn.Linear(1, 1, bias=False, dtype=dtype)
        self.sequential_head = torch.nn.Linear(1, 1, bias=False, dtype=dtype)
        self.backbone.weight.data.fill_(0.7)
        self.sequential_head.weight.data.fill_(0.9)
        self.target_layer_ids = [1]

    def forward(self, values):
        return self.sequential_head(self.backbone(values))


class Tracker:
    def __init__(self):
        self.events = []

    def log(self, values, step):
        self.events.append((step, copy.deepcopy(values)))


class ToyRecipe:
    """Keep production phase/freeze policies, replacing only the objective."""

    def __init__(self, base, random=False):
        self.base, self.random = base, random
        self.algorithm, self.modes = base.algorithm, base.modes
        self.phase_barrier = False  # no real distributed process group in these tests
        self.calls, self.phases = [], []

    def enter_phase(self, session, phase):
        self.base.enter_phase(session, phase)
        self.phases.append(
            (
                phase.name,
                session.draft.sequential_head.weight.requires_grad,
                session.teacher is not None,
            )
        )

    def compute_step(
        self, session, data, *, microbatch_start, microbatch_end, collect_detailed_metrics
    ):
        self.calls.append(
            (
                session.phase.name,
                session.state.global_step,
                session.state.optimizer_step,
                microbatch_start,
                microbatch_end,
                collect_detailed_metrics,
            )
        )
        x, y, weight = data[microbatch_start:microbatch_end].T
        if self.random:
            x = x + torch.rand_like(x)
        values = session.draft(x[:, None]).flatten()
        numerator = ((values - y).square() * weight).sum()
        denominator = weight.sum()
        return TrainingOutput(
            numerator / denominator, RatioTerms(numerator, denominator)
        )

    def log_metrics(self, session, reduced):
        return reduced

    def detail_metrics(self, session):
        return {}

    def training_metadata(self, session):
        return {"loss_config": {"toy": True}}


def make_session(
    algorithm="dflash",
    mode=None,
    *,
    disaggregate=False,
    micro=1,
    random=False,
    use_muon=False,
):
    base = get_training_adapter(algorithm)
    args = SimpleNamespace(
        algorithm=algorithm,
        mode=mode,
        num_epochs=2,
        stage1_epochs=2,
        stage2_epochs=1,
        disaggregate=disaggregate,
        max_steps=None,
        final_save=True,
        accumulation_steps=4,
        log_interval=2,
        save_interval=4,
        seed=17,
        batch_size=4,
        node_batch_size=4,
        draft_ranks_per_node=1,
        draft_micro_batch_size=micro,
        profile=False,
        tp_size=1,
        shard_draft_by_tp=False,
        train_hidden_states_path=None,
        train_data_path="train.jsonl",
        target_model_path="target",
        learning_rate=0.01,
        warmup_ratio=0.0,
        target_ranks_per_node=1,
        target_tp_size=1,
        sglang_ep_size=1,
        use_muon=use_muon,
        **{f"muon_{key}": MUON_DEFAULTS[key] for key in MUON_CLI_KEYS},
    )
    session = TrainingSession(args, ToyRecipe(base, random=random), base.phases(args))
    session.draft = ToyDraft()
    session.teacher = ToyDraft() if mode == "two_stage" else None
    session.teacher_online = session.teacher
    session.steps_per_epoch = 3
    session.total_steps = sum(math.ceil(p.batch_count(3) / 4) for p in session.phases)
    if disaggregate:
        session.topology = SimpleNamespace(
            draft_group=None, draft_global_ranks=[0], nnodes=1
        )
    session.optimizer = BF16Optimizer(
        session.draft,
        parameters=base.optimizer_parameters(session),
        lr=0.01,
        total_steps=session.total_steps,
        warmup_ratio=0,
        max_grad_norm=100,
        muon_config=muon_config_from_args(args),
    )
    session.tracker = Tracker()
    return session


def values(index):
    return torch.tensor(
        [[1.0, 0.1, 1.0], [2.0, 0.5, 3.0], [3.0, 0.6, 2.0], [4.0, 0.3, index + 1.0]]
    )


def source(session, phase):
    for index in range(
        session.state.phase_step, phase.batch_count(session.steps_per_epoch)
    ):
        yield values(index)


def record_save(events):
    def save(session, name):
        assert all(p.grad is None for p in session.draft.parameters())
        events.append((name, copy.deepcopy(session.state)))

    return save


@pytest.mark.parametrize("algorithm,mode", WORKFLOWS)
@pytest.mark.parametrize("disaggregate", [False, True])
@pytest.mark.parametrize("use_muon", [False, True])
def test_all_workflows_execute_one_shared_loop(algorithm, mode, disaggregate, use_muon):
    session = make_session(
        algorithm, mode, disaggregate=disaggregate, use_muon=use_muon
    )
    saves = []
    run_draft(session, batch_source=source, save=record_save(saves))
    assert session.state.optimizer_step == session.total_steps
    expected_batches = sum(p.batch_count(3) for p in session.phases)
    assert session.state.global_step == expected_batches
    assert session.state.phase_index == len(session.phases)
    assert len(session.adapter.calls) == expected_batches * (4 if disaggregate else 1)
    # 3 batches/epoch and accumulation=4: the first update must cross an epoch.
    assert [
        call[2] for call in session.adapter.calls[: 4 if not disaggregate else 16]
    ] == [0] * (4 if not disaggregate else 16)
    assert saves[-1][0] == "final"
    assert saves[-1][1].phase_index == len(session.phases)
    if mode == "two_stage":
        assert session.adapter.phases == [
            ("stage1", False, True),
            ("transition", True, True),
            ("stage2", True, False),
        ]
        boundary = next(state for name, state in saves if name == "stage1/final")
        assert boundary.phase_index == 1 and boundary.phase_step == 0
        assert boundary.global_step == 6
    reports = [(step, p) for step, p in session.tracker.events if "train/loss" in p]
    assert reports
    assert all("train/learning_rate" in p for _, p in reports)


@pytest.mark.parametrize("algorithm,mode", WORKFLOWS)
def test_packet_subdivision_matches_equivalent_large_batch(algorithm, mode):
    sessions = [
        make_session(algorithm, mode, disaggregate=True, micro=m) for m in (1, 2, 4)
    ]
    for session in sessions:
        run_draft(session, batch_source=source, save=lambda *a: None)
    for session in sessions[1:]:
        assert session.state == sessions[0].state
        assert (
            session.optimizer.get_learning_rate()
            == sessions[0].optimizer.get_learning_rate()
        )
        for actual, expected in zip(
            session.draft.parameters(), sessions[0].draft.parameters()
        ):
            torch.testing.assert_close(actual, expected)
        actual_logs = [
            p["train/loss"] for _, p in session.tracker.events if "train/loss" in p
        ]
        expected_logs = [
            p["train/loss"] for _, p in sessions[0].tracker.events if "train/loss" in p
        ]
        assert actual_logs == pytest.approx(expected_logs)


def test_actual_epoch_source_resumes_and_does_not_flush_at_epoch_boundary():
    session = make_session()
    epochs = []

    class Loader:
        sampler = SimpleNamespace(set_epoch=epochs.append)

        def __iter__(self):
            return iter([values(i) for i in range(3)])

    session.dataloader = Loader()
    run_draft(session, save=lambda *a: None)
    assert epochs == [0, 1]
    assert [call[2] for call in session.adapter.calls] == [0, 0, 0, 0, 1, 1]
    assert session.state.optimizer_step == 2


def test_max_steps_tail_and_no_final_save():
    session = make_session("dlite", "teacher")
    session.args.max_steps = 5
    session.args.final_save = False
    session.phases = session.adapter.base.phases(session.args)
    saves = []
    run_draft(session, batch_source=source, save=record_save(saves))
    assert session.state.global_step == 5
    assert session.state.optimizer_step == 2
    assert len(saves) == 1 and "step_4" in saves[0][0]


@pytest.mark.parametrize(
    "at",
    ["stage1/epoch_1_step_4", "stage1/final", "transition", "stage2/epoch_0_step_2"],
)
@pytest.mark.parametrize("use_muon", [False, True])
def test_resume_preserves_random_stream_and_continuous_optimizer(
    tmp_path, at, use_muon
):
    # Capture both within-stage updates and transition boundary checkpoints.
    def make():
        s = make_session("dlite", "two_stage", random=True, use_muon=use_muon)
        s.args.save_interval = 1
        s.args.accumulation_steps = 2
        s.total_steps = sum(math.ceil(p.batch_count(3) / 2) for p in s.phases)
        s.optimizer = BF16Optimizer(
            s.draft,
            parameters=s.adapter.base.optimizer_parameters(s),
            lr=0.01,
            total_steps=s.total_steps,
            warmup_ratio=0,
            max_grad_norm=100,
            muon_config=muon_config_from_args(s.args),
        )
        return s

    session = make()
    initial_rng = capture_rng_state()
    saved = False

    def save(s, name):
        nonlocal saved
        if name != at:
            return
        saved = True
        state = {
            **session_metadata(s),
            **training_identity(s.args),
            **s.optimizer.state_dict(),
            "rng_state": capture_rng_state(),
        }
        save_distributed_training_state(str(tmp_path), state)

    run_draft(session, batch_source=source, save=save)
    assert saved
    expected = copy.deepcopy(session.draft.state_dict())
    restore_rng_state(initial_rng)
    resumed = make()
    resumed.resume_state = load_distributed_training_state(str(tmp_path))
    validate_training_identity(resumed.resume_state, resumed.args)
    validate_session_resume(resumed)
    resumed.optimizer.load_state_dict(resumed.resume_state)
    torch.rand(19)  # setup/replay must not perturb the restored training stream
    run_draft(resumed, batch_source=source, save=lambda *a: None)
    assert resumed.state == session.state
    for name, value in resumed.draft.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert (
        resumed.optimizer.get_learning_rate() == session.optimizer.get_learning_rate()
    )
    common = torch.load(tmp_path / "training_state.pt", weights_only=False)
    assert (
        not {"rng_state", "fp32_master_params", "optimizer_state_dict"} & common.keys()
    )


def test_bf16_resume_restores_unrounded_master_weights():
    model = ToyDraft(torch.bfloat16)
    opt = BF16Optimizer(model, lr=0.00037, total_steps=10)
    x = torch.ones(2, 1, dtype=torch.bfloat16)
    model(x).float().square().sum().backward()
    opt.step()
    state = copy.deepcopy(opt.state_dict())
    restored_model = copy.deepcopy(model)
    restored = BF16Optimizer(restored_model, lr=0.00037, total_steps=10)
    restored.load_state_dict(state)
    assert any(
        not torch.equal(master, p.float())
        for master, p in zip(opt.fp32_params, model.parameters())
    )
    for _ in range(3):
        for m, o in ((model, opt), (restored_model, restored)):
            m(x).float().square().sum().backward()
            o.step()
    for a, b in zip(opt.fp32_params, restored.fp32_params):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_transition_keeps_existing_single_batch_policies():
    assert _cosine_transition_scales(0, 1) == (0.5, 0.5)
    assert _cosine_transition_scales(0, 1, disaggregated=True) == (1.0, 0.0)
    for packet in (False, True):
        assert _cosine_transition_scales(0, 3, disaggregated=packet) == (1.0, 0.0)
        assert _cosine_transition_scales(2, 3, disaggregated=packet) == (0.0, 1.0)


def test_teacher_max_steps_preserves_path_specific_scheduler_length():
    from litespec.training.optimizer import optimizer_step_count

    session = make_session("dlite", "teacher")
    session.args.max_steps = 1
    base = session.adapter.base
    assert optimizer_step_count(base.phases(session.args), 3, 4) == 2
    session.args.disaggregate = True
    assert optimizer_step_count(base.phases(session.args), 3, 4) == 1


def test_v1_is_rejected_and_cursor_contract_is_strict():
    session = make_session()
    with pytest.raises(ValueError, match="current LiteSpec"):
        validate_training_identity({"training_state_format": "litespec_training_v1"})
    for key, value in (
        ("accumulation_steps", 8),
        ("execution_mode", "disaggregate"),
        ("batch_size", 8),
    ):
        state = {**training_identity(session.args), key: value}
        with pytest.raises(ValueError, match=key):
            validate_training_identity(state, session.args)
    session.resume_state = {
        **session_metadata(session),
        "cursor": asdict(RunState(next_batch=1, phase_step=1, global_step=1)),
    }
    with pytest.raises(ValueError, match="optimizer boundary"):
        validate_session_resume(session)


def test_recipes_do_not_own_training_loops_or_side_effects():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "litespec/training/recipes").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in ("backward", "step", "log"), (
                    path,
                    node.lineno,
                )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in (
                    "save_checkpoint",
                    "save_session",
                    "iter_epoch",
                )
        assert not any(isinstance(n, ast.While) for n in ast.walk(tree))
    assert not (root / "litespec/training/protocol.py").exists()
    assert not (root / "litespec/distributed.py").exists()


def test_distributed_public_api_owns_one_group_state():
    import litespec.distributed as public
    from litespec.distributed import collectives, groups

    assert public.get_tp_group is groups.get_tp_group
    assert public.shard_tensor is collectives.shard_tensor
    assert collectives.get_tp_group is groups.get_tp_group


@pytest.mark.parametrize("algorithm,mode", WORKFLOWS)
def test_target_lifecycle_drains_schemas_and_skips_consumed_packets(
    monkeypatch, algorithm, mode
):
    from litespec.distributed import transport
    from litespec.training import runner, setup

    session = make_session(algorithm, mode, disaggregate=True)
    session.args.pipeline_depth = 2
    session.topology.is_target_leader = True
    session.topology.rank = 0
    session.resume_state = {"cursor": {"global_step": 2}}
    events = []

    class Loader:
        sampler = SimpleNamespace(
            set_epoch=lambda epoch: events.append(("epoch", epoch))
        )

        def __iter__(self):
            return iter(range(3))

    class Transport:
        def __init__(self, **kwargs):
            pass

        def send(self, packet, **kwargs):
            events.append(("send", kwargs["batch_id"]))
            return kwargs["batch_id"]

        def wait_send(self, handle):
            events.append(("drain", handle))
            return 0

    base = session.adapter.base
    session.adapter = SimpleNamespace(
        phase_barrier=base.phase_barrier,
        replay_skipped=base.replay_skipped,
        target_epoch_offset=base.target_epoch_offset,
        packet_spec=lambda s, p: SimpleNamespace(schema_id=p.phase_id),
        produce_packet=lambda s, p, values, current: events.append(
            ("produce", current)
        ),
    )
    session.dataloader = Loader()
    monkeypatch.setattr(transport, "NodePacketTransport", Transport)
    monkeypatch.setattr(setup, "_broadcast_input", lambda *a: a[-1])
    monkeypatch.setattr(
        torch.distributed, "barrier", lambda: events.append(("barrier", None))
    )
    runner.run_target(session)
    count = sum(p.batch_count(3) for p in session.phases)
    assert [i for name, i in events if name == "send"] == list(range(2, count))
    assert sorted(i for name, i in events if name == "drain") == list(range(2, count))
    assert [i for name, i in events if name == "produce"] == list(
        range(0 if base.replay_skipped else 2, count)
    )
    assert sum(name == "barrier" for name, _ in events) == (
        len(session.phases) if base.phase_barrier else 0
    )


@pytest.mark.parametrize(
    "phase_name", ["sft", "teacher", "stage1", "transition", "stage2"]
)
@pytest.mark.parametrize("disaggregate", [False, True])
@pytest.mark.parametrize("collect_detailed_metrics", [False, True])
def test_real_dlite_recipe_objectives_on_cpu(
    monkeypatch, phase_name, disaggregate, collect_detailed_metrics
):
    """Real preparation/heads/losses; replace only GPU backbone attention."""
    from refactor_baseline import capture

    from litespec.core.dlite import (
        OnlineDLiteModel,
        prepare_history_hidden_states,
        prepare_target_hidden,
    )
    from litespec.distributed.transport import DLiteBatchPacket
    from litespec.modeling.draft.dlite import DLiteDraftModel
    from litespec.training.recipes import dlite

    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, **kwargs: tensor)
    student = capture(models_only=True)["dlite"]
    config = copy.deepcopy(student.config)
    config.dlite_config.update(
        model_role="swa_teacher", swa_window_size=4, history_layer_ids=[0, 2, 3]
    )
    teacher = DLiteDraftModel(config)
    mode = phase_name if phase_name in ("sft", "teacher") else "two_stage"
    base = get_training_adapter("dlite")
    argv = [
        "--target-model-path",
        "target",
        "--train-data-path",
        "data",
        "--output-dir",
        "unused",
        "--num-anchors",
        "2",
        "--block-size",
        "4",
        "--max-length",
        "12",
    ]
    if mode == "two_stage":
        argv += [
            "--teacher-draft-path",
            "teacher",
            "--stage1-epochs",
            "1",
            "--stage2-epochs",
            "1",
            "--learning-rate",
            ".01",
            "--warmup-ratio",
            "0",
        ]
    args = base.parse_args(mode, argv)
    args.disaggregate = disaggregate
    args.mask_token_id = 31
    session = TrainingSession(args, base, base.phases(args))
    session.steps_per_epoch = 1
    session.state.phase_index = next(
        i for i, p in enumerate(session.phases) if p.name == phase_name
    )
    draft = teacher if mode == "teacher" else student
    monkeypatch.setattr(
        dlite,
        "_initialize_drafts",
        lambda *a: (draft, teacher if mode == "two_stage" else None, None),
    )
    base.initialize(
        session
    )  # also validates teacher-only SWA is not compared to the student
    components = SimpleNamespace(
        lm_head=torch.nn.Linear(16, 32, bias=False),
        embed_tokens=torch.nn.Embedding(32, 16),
    )
    components.lm_head.requires_grad_(False)
    components.embed_tokens.requires_grad_(False)
    session.components = components
    if disaggregate:
        session.topology = SimpleNamespace(draft_group=None)
    session.draft_online = base.make_online(session)
    session.fsdp_model = session.draft_online
    base.enter_phase(session, session.phase)
    anchors = torch.tensor([[0, 4]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    monkeypatch.setattr(
        OnlineDLiteModel, "sample_anchor_positions", lambda *a: (anchors, keep)
    )

    def backbone(wrapper, batch, *, seq_len):
        # Exercise trainable model parameters and real sequential heads without CUDA kernels.
        scale = next(wrapper.draft_model.layers.parameters()).flatten()[0]
        hidden = batch.query_embeddings[:, :, -3:, :] * scale
        if batch.shared_fused_history is not None:
            hidden = hidden + batch.shared_fused_history.mean(dim=1)[:, None, None, :]
        return hidden

    monkeypatch.setattr(OnlineDLiteModel, "forward_backbone", backbone)
    ids = torch.arange(12).view(1, -1)
    hidden = {i: torch.randn(1, 12, 16) for i in range(4)}
    logits = components.lm_head(hidden[3])
    session.target = SimpleNamespace(
        generate_training_data=lambda *a, **k: SimpleNamespace(
            hidden_states=hidden, logits=logits
        )
    )
    if disaggregate:
        session.packet_metadata = base.packet_metadata(session)
        history = (
            prepare_history_hidden_states(
                hidden, session.packet_metadata["history_layer_ids"], 4
            )
            if session.phase.need_history
            else None
        )
        prediction = (
            dlite._prediction_hidden(hidden[3], anchors, 4)
            if session.phase.need_prediction_hidden
            else None
        )
        batch = DLiteBatchPacket(
            ids,
            torch.ones_like(ids),
            anchors,
            keep,
            prepare_target_hidden(
                hidden, anchors, session.packet_metadata["target_layer_ids"], 4
            ),
            history,
            prediction,
        )
        output = base.compute_step(
            session, batch, microbatch_start=0, microbatch_end=1,
            collect_detailed_metrics=collect_detailed_metrics,
        )
    else:
        output = base.compute_step(
            session,
            dict(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                loss_mask=torch.ones_like(ids),
            ),
            collect_detailed_metrics=collect_detailed_metrics,
        )
    assert isinstance(output, TrainingOutput)
    assert torch.isfinite(output.loss)
    output.loss_terms.numerator.backward()
    assert any(p.grad is not None for p in draft.parameters())
    if phase_name == "stage1":
        assert "expected_accept_len" not in output.ratio_metrics
        assert all(p.grad is None for p in draft.sequential_head.parameters())
    elif collect_detailed_metrics:
        assert "expected_accept_len" in output.ratio_metrics
    if not collect_detailed_metrics:
        assert not output.ratio_metrics


@pytest.mark.parametrize("algorithm", ["dflash", "dflash2", "dspark"])
def test_real_dflash_family_recipe_online_offline_and_packet_match(monkeypatch, algorithm):
    from refactor_baseline import capture

    from litespec.distributed.transport import DFlashFamilyBatchPacket

    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, **kwargs: tensor)
    base = get_training_adapter(algorithm)
    args = base.parse_args(
        argv=[
            "--target-model-path",
            "target",
            "--train-data-path",
            "data",
            "--output-dir",
            "unused",
            "--attention-backend",
            "eager",
            "--num-anchors",
            "2",
            "--block-size",
            "4",
            "--max-length",
            "12",
        ]
    )
    session = TrainingSession(args, base, base.phases(args))
    session.draft = capture(models_only=True)[algorithm]
    args.mask_token_id = 31
    session.components = SimpleNamespace(
        lm_head=torch.nn.Linear(16, 32, bias=False),
        embed_tokens=torch.nn.Embedding(32, 16),
    )
    session.components.lm_head.requires_grad_(False)
    session.components.embed_tokens.requires_grad_(False)
    session.draft_online = base.make_online(session)
    session.fsdp_model = session.draft_online
    session.total_steps = 10
    hidden = {i: torch.randn(1, 12, 16) for i in range(4)}
    data = dict(
        input_ids=torch.arange(12).view(1, -1),
        attention_mask=torch.ones(1, 12),
        loss_mask=torch.ones(1, 12),
        hidden_states=hidden,
    )
    session.target = SimpleNamespace(
        generate_training_data=lambda *a, **k: SimpleNamespace(hidden_states=hidden)
    )
    results = []
    for route in ("online", "offline", "packet"):
        torch.manual_seed(7)
        session.draft.zero_grad(set_to_none=True)
        if route == "online":
            output = base.compute_step(session, data, collect_detailed_metrics=True)
        elif route == "offline":
            session.target = None
            output = base.compute_step(session, data, collect_detailed_metrics=True)
        else:
            session.topology = SimpleNamespace(draft_group=None)
            packet = DFlashFamilyBatchPacket(
                data["input_ids"], data["loss_mask"], hidden[1], hidden[3]
            )
            output = base.compute_step(
                session,
                packet,
                microbatch_start=0,
                microbatch_end=1,
                collect_detailed_metrics=True,
            )
        output.loss_terms.numerator.backward()
        results.append(
            (
                output.loss.detach(),
                {
                    n: p.grad.clone()
                    for n, p in session.draft.named_parameters()
                    if p.grad is not None
                },
            )
        )
    for loss, grads in results[1:]:
        torch.testing.assert_close(loss, results[0][0], rtol=0, atol=0)
        assert grads.keys() == results[0][1].keys()
        for key in grads:
            torch.testing.assert_close(grads[key], results[0][1][key], rtol=0, atol=0)


def _runner_rank_worker(rank, rendezvous):
    """Simulate FSDP rank-averaged backward using real two-rank Gloo collectives."""
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        session = make_session()
        parameter = torch.nn.Parameter(torch.tensor(0.2))
        reference = torch.tensor(0.2, requires_grad=True)
        expected = []
        # Six logical batches: four in the first window and two in the tail.
        for begin, end in ((0, 4), (4, 6)):
            numerator, denominator = 0.0, 0.0
            for other in range(2):
                for i in range(begin, end):
                    weight, x = float(other + i + 1), float(i + 1)
                    numerator = numerator + (reference * x).square() * weight
                    denominator += weight
            (grad,) = torch.autograd.grad(numerator / denominator, reference)
            expected.append(grad.item())
            reference = (reference - 0.01 * grad).detach().requires_grad_(True)

        def rank_average(gradient):
            dist.all_reduce(gradient)
            return gradient / 2

        parameter.register_hook(rank_average)

        class Optimizer:
            updates = 0

            def scale_model_gradients(self, factor):
                parameter.grad.mul_(factor)

            def step(self):
                assert parameter.grad.item() == pytest.approx(expected[self.updates])
                norm = parameter.grad.abs().item()
                with torch.no_grad():
                    parameter.add_(parameter.grad, alpha=-0.01)
                parameter.grad = None
                self.updates += 1
                return norm

            def get_learning_rate(self):
                return 0.01

        def compute(s, data, **kwargs):
            index = s.state.global_step
            weight = torch.tensor(float(rank + index + 1))
            numerator = (parameter * (index + 1)).square() * weight
            return TrainingOutput(numerator / weight, RatioTerms(numerator, weight))

        session.optimizer = Optimizer()
        session.adapter.compute_step = compute
        run_draft(session, batch_source=source, save=lambda *args: None)
        torch.testing.assert_close(parameter, reference)
        assert session.state.optimizer_step == 2
    finally:
        dist.destroy_process_group()


def test_full_runner_two_rank_unequal_weights_and_tail(tmp_path):
    torch.multiprocessing.spawn(
        _runner_rank_worker, args=((tmp_path / "runner-rendezvous").as_uri(),), nprocs=2
    )
