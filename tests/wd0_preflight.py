"""Validate the four zero-decay configurations and smoke-test the new baseline."""
import json
import os
import subprocess
import sys
from pathlib import Path
import yaml
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from scripts.config.launch_dflash_family import load_config
from scripts.config.training_launcher import arguments_to_cli
from litespec.training.algorithms import get_training_adapter
from litespec.training.muon import muon_config_from_args
manifest = json.loads((root / "tests/wd0_experiments.json").read_text())
audit = []
for item in manifest["experiments"]:
    config = yaml.safe_load(Path(item["config"]).read_text())
    env, cli = load_config(Path(item["config"]), item["algorithm"])
    args = get_training_adapter(item["algorithm"]).parse_args(argv=cli + [
        "--target-model-path", config["launcher"]["target_model"],
        "--train-data-path", config["launcher"]["train_data"],
        "--output-dir", config["launcher"]["output_dir"]])
    muon = muon_config_from_args(args)
    assert args.weight_decay == muon["weight_decay"] == muon["adamw_weight_decay"] == 0.0
    assert args.resume_from is None and args.init_from is None
    assert args.block_size == 8 and args.num_draft_layers == 5 and args.seed == 42 and args.num_epochs == 6
    assert args.batch_size == args.accumulation_steps == args.tp_size == 1
    assert args.learning_rate == 0.0006 and muon["lr"] == 0.02 and muon["adamw_lr"] == 0.0004
    if item["algorithm"] == "realsemi":
        assert args.num_parallel_layers == 4 and args.ce_loss_alpha == 0.1 and args.l1_loss_alpha == 0.9
    audit.append({"name": item["name"], "weight_decay": args.weight_decay,
                  "muon_weight_decay": muon["weight_decay"], "muon_adamw_weight_decay": muon["adamw_weight_decay"],
                  "resume_from": args.resume_from, "init_from": args.init_from})
(root / "tests/wd0_config_validation.json").write_text(json.dumps(audit, indent=2) + "\n")
print("ALL_FOUR_CONFIGS_VALIDATED", json.dumps(audit), flush=True)
item = manifest["experiments"][0]
config = yaml.safe_load(Path(item["config"]).read_text())
args = dict(config["arguments"])
args.update(report_to="none", log_interval=1)
args.pop("wandb_run_id", None)
command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "8",
           "--master_port", "29615", "tests/wd0_training_smoke.py", "dflash",
           *arguments_to_cli(args), "--target-model-path", config["launcher"]["target_model"],
           "--train-data-path", config["launcher"]["train_data"],
           "--output-dir", str(root / "tests/smoke_outputs/dflash-5l-wd0-8gpu")]
log = root / "tests/smoke-dflash-5l-wd0-8gpu.log"
with log.open("w") as stream:
    result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
summary = {"name": "dflash-5l-wd0-8gpu", "exit_code": result.returncode, "log": str(log)}
(root / "tests/wd0_smoke_result.json").write_text(json.dumps(summary, indent=2) + "\n")
print("WD0_SMOKE_RESULT", json.dumps(summary), flush=True)
raise SystemExit(result.returncode)
