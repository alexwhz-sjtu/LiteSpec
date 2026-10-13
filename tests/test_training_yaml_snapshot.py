"""Launch-time YAML survives source edits and is saved by the model writer."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from litespec.training import algorithms, checkpoint
from scripts.config import launch_dflash_family, launch_dlite, training_launcher


@pytest.fixture
def checkpoint_session(monkeypatch, tmp_path):
    # Exercise the real save_session -> save_checkpoint path without GPUs or
    # model weights; the writer is global rank 4, draft-group rank 0.
    adapter = SimpleNamespace(
        parse_args=lambda *args, **kwargs: None,
        checkpoint_metadata=lambda model: {},
    )
    monkeypatch.setattr(algorithms, "get_training_adapter", lambda name: adapter)
    monkeypatch.setattr(checkpoint, "get_training_adapter", lambda name: adapter)
    monkeypatch.setattr(checkpoint, "session_metadata", lambda session: {})
    monkeypatch.setattr(checkpoint, "training_identity", lambda args: {})
    monkeypatch.setattr(checkpoint, "capture_rng_state", lambda: {})
    monkeypatch.setattr(
        checkpoint, "save_distributed_training_state", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(checkpoint.FSDP, "state_dict_type", lambda *args: nullcontext())
    monkeypatch.setattr(
        checkpoint.dist, "get_rank", lambda group=None: 4 if group is None else 0
    )
    monkeypatch.setattr(checkpoint.dist, "barrier", lambda **kwargs: None)

    def export(model, save_dir, **kwargs):
        Path(save_dir, "model.stub").write_text("exported")

    monkeypatch.setattr(checkpoint, "export_draft", export)

    def make_session(algorithm="dflash"):
        return algorithms.TrainingSession(
            args=SimpleNamespace(output_dir=str(tmp_path / "output")),
            adapter=adapter,
            phases=[],
            draft=SimpleNamespace(method_config_key=f"{algorithm}_config"),
            fsdp_model=SimpleNamespace(state_dict=lambda: {}),
            optimizer=SimpleNamespace(state_dict=lambda: {}),
            topology=SimpleNamespace(draft_group=object(), draft_global_ranks=[4]),
        )

    return make_session


@pytest.mark.parametrize(
    "algorithm,mode",
    [
        ("dlite", "sft"),
        ("dlite", "teacher"),
        ("dlite", "two_stage"),
        ("dflash", None),
        ("dflash2", None),
        ("dspark", None),
    ],
)
def test_yaml_snapshot_from_launcher_to_every_checkpoint(
    monkeypatch, tmp_path, checkpoint_session, algorithm, mode
):
    identity = f"mode: {mode}" if mode else f"algorithm: {algorithm}"
    source = (
        f"# 启动时的配置，保留注释和 CRLF\n{identity}\n"
        "launcher:\n"
        "  target_model: /model\n"
        "  train_data: /data.jsonl\n"
        "  output_dir: /output\n"
        "arguments:\n"
        "  learning_rate: 0.0004\n"
    )
    if mode == "two_stage":
        source += "  stage1_epochs: 1\n  stage2_epochs: 1\n"
    source = source.replace("\n", "\r\n").encode("utf-8")
    config_path = tmp_path / "experiment.yaml"
    config_path.write_bytes(source)
    launcher = launch_dlite if mode else launch_dflash_family
    env, cli = launcher.load_config(config_path, mode or algorithm)

    # Editing even before torchrun starts must not change the already parsed
    # snapshot. CLI overrides also do not rewrite the original YAML.
    config_path.write_text("replaced after parsing: true\n")
    monkeypatch.setenv("LITESPEC_TRAINING_YAML", "stale inherited snapshot")
    monkeypatch.delenv("DRY_RUN", raising=False)
    monkeypatch.setattr(training_launcher.os, "chdir", lambda path: None)
    child_env = {}
    monkeypatch.setattr(
        training_launcher.os,
        "execvpe",
        lambda executable, command, environment: child_env.update(environment),
    )
    passthrough = ["--learning-rate", "0.0008"]
    if mode == "two_stage":
        passthrough.extend(["--teacher-draft-path", "/teacher"])
    training_launcher.launch_training(
        label=mode or algorithm,
        module=launch_dlite.SCRIPTS[mode] if mode else f"scripts.{algorithm}.train",
        configured_env=env,
        configured_cli=cli,
        passthrough=passthrough,
        project_dir=Path(__file__).resolve().parents[1],
    )
    assert child_env["LITESPEC_TRAINING_YAML"].encode("utf-8") == source

    # Emulate torchrun's inherited environment, then freeze it in the session.
    monkeypatch.setenv("LITESPEC_TRAINING_YAML", child_env["LITESPEC_TRAINING_YAML"])
    session = checkpoint_session(algorithm)
    monkeypatch.setenv("LITESPEC_TRAINING_YAML", "changed during training")
    config_path.unlink()
    for name in ("step_1", "final"):
        save_dir = Path(checkpoint.save_session(session, name))
        assert (save_dir / "model.stub").is_file()
        assert (save_dir / "training_config.yaml").read_bytes() == source


def test_direct_cli_has_no_source_yaml(monkeypatch, checkpoint_session):
    monkeypatch.delenv("LITESPEC_TRAINING_YAML", raising=False)
    session = checkpoint_session()
    save_dir = Path(checkpoint.save_session(session, "final"))
    assert (save_dir / "model.stub").is_file()
    assert not (save_dir / "training_config.yaml").exists()


def test_non_writer_rank_does_not_export_snapshot(monkeypatch, checkpoint_session):
    monkeypatch.setenv("LITESPEC_TRAINING_YAML", "algorithm: dflash\n")
    session = checkpoint_session()
    monkeypatch.setattr(
        checkpoint.dist, "get_rank", lambda group=None: 5 if group is None else 1
    )
    save_dir = Path(checkpoint.save_session(session, "final"))
    assert not (save_dir / "training_config.yaml").exists()
    assert not (save_dir / "model.stub").exists()
