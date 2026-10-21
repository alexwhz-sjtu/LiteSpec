"""Generate isolated, reproducible experiment configurations from DFlash2 defaults."""

import copy
import hashlib
import json
from pathlib import Path
import yaml

root = Path(__file__).resolve().parents[1]
base = yaml.safe_load((root / "scripts/config/dflash2.yaml").read_text())
group = "realsemi-qwen3-4b-20261020-v1"
experiments = []
for name, mode in (
    ("dflash2-5l", None),
    ("realsemi-4p1s-full", "full"),
    ("realsemi-4p1s-last", "last"),
):
    config = copy.deepcopy(base)
    config["algorithm"] = "dflash2" if mode is None else "realsemi"
    config["launcher"].pop("train_hidden_states", None)
    config["launcher"].update(
        target_model="/data/temp/artifacts/models/Qwen3-4B",
        train_data="/data/temp/artifacts/training_data/open_perfectblend_80k_qwen3_4b.jsonl",
        output_dir=str(root / "outputs" / group / name),
    )
    config["arguments"].update(
        wandb_name=name,
        wandb_run_id=hashlib.sha256((group + name).encode()).hexdigest()[:8],
    )
    if mode:
        for field in (
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
            "selector_loss_alpha",
        ):
            config["arguments"].pop(field, None)
        config["arguments"].update(
            num_parallel_layers=4,
            serial_context_mode=mode,
            ce_loss_alpha=0.1,
            l1_loss_alpha=0.9,
        )
    path = root / "scripts/config" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    experiments.append(
        dict(
            name=name,
            algorithm=config["algorithm"],
            config=str(path),
            output_dir=config["launcher"]["output_dir"],
            wandb_run_id=config["arguments"]["wandb_run_id"],
        )
    )
    if mode == "full":
        default = copy.deepcopy(config)
        default["launcher"]["train_data"] = "/path/to/train.jsonl"
        default["launcher"]["output_dir"] = "./outputs/realsemi_qwen3_4b"
        default["arguments"].pop("wandb_run_id")
        default["arguments"]["wandb_name"] = "realsemi"
        (root / "scripts/config/realsemi.yaml").write_text(
            yaml.safe_dump(default, sort_keys=False)
        )
manifest = dict(
    group=group,
    project="spec-decoding",
    experiments=experiments,
    data_path=base.get(
        "data_path",
        "/data/temp/artifacts/training_data/open_perfectblend_80k_qwen3_4b.jsonl",
    ),
)
(root / "tests/realsemi_experiments.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
