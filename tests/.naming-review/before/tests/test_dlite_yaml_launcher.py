import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "scripts" / "config" / "launch_dlite.py"
SPEC = importlib.util.spec_from_file_location("launch_dlite", LAUNCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.mark.parametrize(
    ("mode", "filename", "port"),
    [
        ("sft", "dlite_sft.yaml", "29503"),
        ("teacher", "dlite_teacher.yaml", "29504"),
        ("two_stage", "dlite_two_stage.yaml", "29505"),
    ],
)
def test_repository_dlite_yaml_loads(mode, filename, port):
    env, cli = launcher.load_config(ROOT / "scripts/config" / filename, mode)

    assert env["TARGET_MODEL"] == "Qwen/Qwen3-4B"
    assert env["TRAIN_DATA_PATH"] == "/path/to/train.jsonl"
    assert env["NPROC_PER_NODE"] == "8"
    assert env["MASTER_PORT"] == port
    assert cli[cli.index("--accumulation-steps") + 1] == "1"
    assert cli[cli.index("--wandb-project") + 1] == "dlite-training"


def test_two_stage_emits_training_arguments():
    env, cli = launcher.load_config(
        ROOT / "scripts/config/dlite_two_stage.yaml", "two_stage"
    )

    assert env["TEACHER_DRAFT_PATH"] == "/path/to/dlite_teacher_checkpoint"
    assert cli[cli.index("--stage1-epochs") + 1] == "1"
    assert cli[cli.index("--learning-rate") + 1] == "0.0004"
    assert cli[cli.index("--stage2-epochs") + 1] == "6"


def test_sft_does_not_set_teacher_swa_window():
    _, cli = launcher.load_config(ROOT / "scripts/config/dlite_sft.yaml", "sft")

    assert "--swa-window-size" not in cli


def test_two_stage_validates_teacher_after_cli_overrides(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "mode: two_stage\n"
        "launcher:\n"
        "  target_model: /model\n"
        "  train_data: /data.jsonl\n"
        "  output_dir: /output\n"
        "arguments:\n"
        "  stage1_epochs: 1\n"
        "  stage2_epochs: 1\n"
        "  learning_rate: 0.0004\n",
        encoding="utf-8",
    )

    from litespec.training.algorithms import get_training_adapter
    from scripts.config.training_launcher import training_base_cli

    env, cli = launcher.load_config(path, "two_stage")
    adapter = get_training_adapter("dlite")
    args = [*training_base_cli(env), *cli, "--warmup-ratio", "0.04"]
    with pytest.raises(SystemExit):
        adapter.parse_args("two_stage", argv=args)
    parsed = adapter.parse_args(
        "two_stage", argv=[*args, "--teacher-draft-path", "/teacher"]
    )
    assert parsed.teacher_draft_path == "/teacher"
