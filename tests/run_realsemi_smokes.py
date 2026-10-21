"""Single- and eight-rank checks with production dimensions and two updates."""

import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.config.training_launcher import arguments_to_cli

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "tests/realsemi_experiments.json").read_text())
results = []
for ranks in (1, 8):
    for experiment in manifest["experiments"]:
        config = yaml.safe_load(Path(experiment["config"]).read_text())
        args = dict(config["arguments"])
        args.update(report_to="none", log_interval=1)
        args.pop("wandb_run_id", None)
        label = f"{experiment['name']}-{ranks}gpu"
        output = root / "tests/smoke_outputs" / label
        cli = arguments_to_cli(args) + [
            "--target-model-path",
            config["launcher"]["target_model"],
            "--train-data-path",
            config["launcher"]["train_data"],
            "--output-dir",
            str(output),
        ]
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(ranks),
            "--master_port",
            "29615",
            "tests/realsemi_training_smoke.py",
            experiment["algorithm"],
            *cli,
        ]
        log = root / "tests" / f"smoke-{label}.log"
        print("SMOKE_START", label, flush=True)
        with log.open("w") as stream:
            proc = subprocess.run(
                command, cwd=root, stdout=stream, stderr=subprocess.STDOUT
            )
        results.append(dict(name=label, exit_code=proc.returncode, log=str(log)))
        (root / "tests/realsemi_smoke_results.json").write_text(
            json.dumps(results, indent=2)
        )
        if proc.returncode:
            print("SMOKE_FAILED", label, flush=True)
            raise SystemExit(proc.returncode)
        print("SMOKE_PASSED", label, flush=True)
