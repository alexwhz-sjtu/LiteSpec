import os
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.config.training_launcher import ConfigError, validate_common_settings

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "scripts" / "config"
CONFIGS = (
    "dflash",
    "dflash2",
    "dspark",
    "dlite_sft",
    "dlite_teacher",
    "dlite_two_stage",
)
COMMON_LAUNCHER_KEYS = (
    "target_model",
    "train_data",
    "train_hidden_states",
    "output_dir",
    "nnodes",
    "node_rank",
    "nproc_per_node",
    "master_addr",
    "master_port",
)
COMMON_ARGUMENT_KEYS = (
    "target_model_backend",
    "sglang_attention_backend",
    "sglang_mem_fraction_static",
    "sglang_ep_size",
    "attention_backend",
    "flex_attention_backend",
    "chat_template",
    "max_length",
    "batch_size",
    "accumulation_steps",
    "tp_size",
    "shard_draft_by_tp",
    "dist_timeout",
    "num_anchors",
    "learning_rate",
    "warmup_ratio",
    "max_grad_norm",
    "use_muon",
    "muon_lr",
    "muon_momentum",
    "muon_weight_decay",
    "muon_matrices_per_rank",
    "muon_adamw_lr",
    "muon_adamw_weight_decay",
    "seed",
    "cache_dir",
    "build_dataset_num_proc",
    "dataloader_num_workers",
    "log_interval",
    "save_interval",
    "report_to",
    "wandb_project",
    "wandb_name",
    "disaggregate",
    "target_ranks_per_node",
    "draft_ranks_per_node",
    "target_tp_size",
    "node_batch_size",
    "draft_micro_batch_size",
    "pipeline_depth",
    "profile",
)
SCRIPTS = {
    "dflash": "scripts/dflash/run_training_dflash.sh",
    "dflash2": "scripts/dflash2/run_training_dflash2.sh",
    "dspark": "scripts/dspark/run_training_dspark.sh",
    "dlite_sft": "scripts/dlite/run_training_dlite_sft.sh",
    "dlite_teacher": "scripts/dlite/run_training_dlite_teacher.sh",
    "dlite_two_stage": "scripts/dlite/run_training_dlite_two_stage.sh",
}


def _config(name):
    return yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", CONFIGS)
def test_common_yaml_fields_are_present_and_ordered(name):
    config = _config(name)
    launcher_keys = list(config["launcher"])
    argument_keys = list(config["arguments"])

    data_key = (
        "train_data" if config["launcher"].get("train_data") else "train_hidden_states"
    )
    expected_launcher = [
        "target_model",
        data_key,
        "output_dir",
        "nnodes",
        "node_rank",
        "nproc_per_node",
        "master_addr",
        "master_port",
    ]
    assert [key for key in launcher_keys if key in COMMON_LAUNCHER_KEYS] == (
        expected_launcher
    )
    assert [key for key in argument_keys if key in COMMON_ARGUMENT_KEYS] == list(
        COMMON_ARGUMENT_KEYS
    )


def test_algorithm_specific_yaml_fields_stay_scoped():
    configs = {name: _config(name) for name in CONFIGS}

    assert "swa_window_size" in configs["dlite_teacher"]["arguments"]
    assert all(
        "swa_window_size" not in configs[name]["arguments"]
        for name in CONFIGS
        if name != "dlite_teacher"
    )
    assert "local_position" in configs["dlite_sft"]["arguments"]
    assert "teacher_draft" in configs["dlite_two_stage"]["launcher"]
    assert "selector_rank" in configs["dflash2"]["arguments"]
    assert "markov_rank" in configs["dspark"]["arguments"]
    assert configs["dspark"]["arguments"]["backbone_conv"] is False
    assert "conv_kernel_size" in configs["dspark"]["arguments"]
    assert "conv_group_size" in configs["dspark"]["arguments"]


@pytest.mark.parametrize("name", CONFIGS)
def test_muon_yaml_roundtrip_and_cli_override(name):
    from litespec.training.algorithms import get_training_adapter
    from litespec.training.muon import MUON_CLI_KEYS, MUON_DEFAULTS, muon_config_from_args
    from scripts.config.training_launcher import arguments_to_cli

    config = _config(name)
    args = config["arguments"]
    for key in MUON_CLI_KEYS:
        value = MUON_DEFAULTS[key]
        assert args[f"muon_{key}"] == (
            list(value) if isinstance(value, tuple) else value
        )
    mode = name.removeprefix("dlite_") if name.startswith("dlite_") else None
    adapter = get_training_adapter("dlite" if mode else name)
    base = [
        "--target-model-path",
        "target",
        "--train-data-path",
        "train.jsonl",
        "--output-dir",
        "out",
    ]
    if mode == "two_stage":
        base += ["--teacher-draft-path", "teacher"]
    parsed = adapter.parse_args(
        mode, argv=base + arguments_to_cli(args) + ["--use-muon", "--muon-lr", "0.03"]
    )
    assert muon_config_from_args(parsed)["lr"] == 0.03


@pytest.mark.parametrize(("name", "script"), SCRIPTS.items())
def test_shell_entrypoints_use_default_yaml_and_append_cli_overrides(name, script):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    for config_arg in ([], [str(CONFIG_DIR / f"{name}.yaml")]):
        result = subprocess.run(
            [
                str(ROOT / script),
                *config_arg,
                "--batch-size",
                "7",
                "--no-disaggregate",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

        assert result.stdout.rfind("--batch-size 7") > result.stdout.find(
            "--batch-size"
        )
        assert result.stdout.rstrip().endswith("--no-disaggregate")


def test_cli_can_disable_boolean_enabled_by_yaml(tmp_path):
    config = _config("dflash")
    config["arguments"]["disaggregate"] = True
    path = tmp_path / "dflash.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    env = os.environ.copy()
    env["DRY_RUN"] = "1"

    result = subprocess.run(
        [
            str(ROOT / SCRIPTS["dflash"]),
            str(path),
            "--no-disaggregate",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--disaggregate" in result.stdout
    assert result.stdout.rstrip().endswith("--no-disaggregate")


def _launcher(**overrides):
    values = {
        "target_model": "target",
        "train_data": "train.jsonl",
        "output_dir": "out",
        "nproc_per_node": 8,
        "nnodes": 1,
        "node_rank": 0,
    }
    values.update(overrides)
    return values


def _arguments(**overrides):
    values = {
        "target_model_backend": "sglang",
        "batch_size": 1,
        "tp_size": 1,
        "shard_draft_by_tp": False,
        "disaggregate": False,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ("launcher", "arguments", "message"),
    [
        (
            _launcher(),
            _arguments(shard_draft_by_tp=True),
            "tp_size > 1",
        ),
        (
            _launcher(),
            _arguments(shard_draft_by_tp=True, tp_size=2),
            "batch_size == tp_size",
        ),
        (
            _launcher(train_data=None, train_hidden_states="cache"),
            _arguments(shard_draft_by_tp=True, tp_size=2, batch_size=2),
            "train_data only",
        ),
        (
            _launcher(),
            _arguments(
                shard_draft_by_tp=True,
                tp_size=2,
                batch_size=2,
                disaggregate=True,
            ),
            "incompatible",
        ),
        (
            _launcher(nproc_per_node=8),
            _arguments(
                disaggregate=True,
                target_ranks_per_node=3,
                draft_ranks_per_node=4,
                target_tp_size=1,
                node_batch_size=12,
            ),
            "nproc_per_node",
        ),
    ],
)
def test_shared_launcher_rejects_invalid_parallel_combinations(
    launcher, arguments, message
):
    with pytest.raises(ConfigError, match=message):
        validate_common_settings(launcher, arguments)
