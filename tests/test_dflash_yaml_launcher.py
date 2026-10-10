import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "scripts" / "config" / "launch_dflash_family.py"
SPEC = importlib.util.spec_from_file_location("launch_dflash_family", LAUNCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.mark.parametrize(
    ("algorithm", "filename", "block_size"),
    [
        ("dflash", "dflash.yaml", "16"),
        ("dflash2", "dflash2.yaml", "8"),
        ("dspark", "dspark.yaml", "8"),
    ],
)
def test_repository_family_yaml_loads_expected_values(algorithm, filename, block_size):
    path = ROOT / "scripts/config" / filename
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    env, cli = launcher.load_config(path, algorithm)

    assert env["TARGET_MODEL"] == str(raw["launcher"]["target_model"])
    assert env["NPROC_PER_NODE"] == "8"
    if raw["launcher"].get("train_data"):
        assert env["TRAIN_DATA_PATH"] == str(raw["launcher"]["train_data"])
    else:
        assert env["TRAIN_HIDDEN_STATES_PATH"] == str(
            raw["launcher"]["train_hidden_states"]
        )
    assert cli[cli.index("--max-length") + 1] == "4096"
    assert cli[cli.index("--block-size") + 1] == block_size
    assert "--enable-thinking" not in cli
def test_dflash2_yaml_keeps_selector_top_k_distinct_from_sampling_top_k():
    path = ROOT / "scripts/config/dflash2.yaml"

    _, cli = launcher.load_config(path, "dflash2")

    assert cli[cli.index("--selector-top-k") + 1] == "16"


def test_dspark_yaml_exposes_optional_backbone_convolution():
    path = ROOT / "scripts/config/dspark.yaml"

    _, cli = launcher.load_config(path, "dspark")

    # The repository default leaves the optional convolution disabled, while
    # retaining its tunable shape parameters for a one-line YAML opt-in.
    assert "--backbone-conv" not in cli
    assert cli[cli.index("--conv-kernel-size") + 1] == "2"
    assert cli[cli.index("--conv-group-size") + 1] == "16"


def test_yaml_requires_exactly_one_data_source(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "algorithm: dflash2\n"
        "launcher:\n"
        "  target_model: /model\n"
        "  output_dir: /output\n",
        encoding="utf-8",
    )

    with pytest.raises(launcher.ConfigError, match="exactly one"):
        launcher.load_config(path, "dflash2")
