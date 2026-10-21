"""Durable sequential experiment queue; a failed job prevents subsequent launches."""

import argparse
import fcntl
import hashlib
import yaml
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "tests/wd0_experiments.json").read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--start-from", choices=[item["name"] for item in manifest["experiments"]])
args = parser.parse_args()
start_index = next((i for i, item in enumerate(manifest["experiments"])
                    if item["name"] == args.start_from), 0)
status_path = root / "tests/wd0_queue_status.json"
lock = (root / "tests/wd0_queue.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
smokes = json.loads((root / "tests/realsemi_smoke_results.json").read_text())
if len(smokes) != 6 or any(item["exit_code"] != 0 for item in smokes):
    raise RuntimeError(
        "All six production-dimension smoke checks must pass before training"
    )
if (
    "CHECKPOINT_FSDP_MUON_RESUME_PASSED"
    not in (root / "tests/realsemi_checkpoint.log").read_text()
):
    raise RuntimeError("Distributed checkpoint/resume validation is required")
# Keep the prior experiment queue and its outputs intact.
if not args.start_from and status_path.exists():
    raise RuntimeError("The wd0 queue already has state; refusing a duplicate launch")
validation = json.loads((root / "tests/wd0_config_validation.json").read_text())
if len(validation) != 4 or any(any(item[key] != 0.0 for key in
    ("weight_decay", "muon_weight_decay", "muon_adamw_weight_decay")) for item in validation):
    raise RuntimeError("All four parsed zero-decay configurations must pass validation")
smoke = json.loads((root / "tests/wd0_smoke_result.json").read_text())
if smoke["exit_code"] != 0:
    raise RuntimeError("The new DFlash zero-decay eight-GPU smoke must pass")
for rank in range(8):
    groups = json.loads((root / f"tests/wd0_optimizer_groups_rank{rank}.json").read_text())
    if not groups or any(value != 0.0 for values in groups.values() for value in values):
        raise RuntimeError("Smoke optimizer groups did not use zero weight decay")
for item in manifest["experiments"]:
    path = Path(item["config"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["config_sha256"]:
        raise RuntimeError(f"Configuration changed after validation: {path}")
    config = yaml.safe_load(path.read_text())
    if any(config["arguments"][key] != 0.0 for key in
           ("weight_decay", "muon_weight_decay", "muon_adamw_weight_decay")):
        raise RuntimeError(f"Nonzero weight decay: {path}")
    if not args.start_from and Path(item["output_dir"]).exists():
        raise RuntimeError(f"Refusing to reuse output: {item['output_dir']}")

state = dict(
    pid=os.getpid(),
    group=manifest["group"],
    status="running",
    experiments=[dict(item, status="queued") for item in manifest["experiments"]],
)

if args.start_from:
    previous = json.loads(status_path.read_text())
    assert previous["status"] == "stopped", "Stop the previous queue before switching"
    history = {item["name"]: item for item in previous["experiments"]}
    for index, item in enumerate(state["experiments"]):
        if index < start_index:
            state["experiments"][index] = history[item["name"]]
        else:
            assert history[item["name"]]["status"] == "queued", "Refusing to restart an existing run"
    state["start_from"] = args.start_from
    state["previous_queue_pid"] = previous["pid"]


def save():
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(status_path)


os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_ENTITY"] = manifest["entity"]
os.environ["WANDB_RUN_GROUP"] = manifest["group"]
os.environ.setdefault("OMP_NUM_THREADS", "4")
save()
for item in state["experiments"][start_index:]:
    name = item["name"]
    if hashlib.sha256(Path(item["config"]).read_bytes()).hexdigest() != item["config_sha256"]:
        state["status"] = "failed"
        item["status"] = "failed"
        item["error"] = "configuration changed while queued"
        save()
        raise RuntimeError(item["error"])
    item["status"] = "running"
    item["started_at"] = datetime.now(timezone.utc).isoformat()
    item["log"] = str(root / "tests" / f"train-{name}.log")
    save()
    print("TRAIN_START", name, flush=True)
    with open(item["log"], "a") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "scripts/config/launch_dflash_family.py",
                item["algorithm"],
                item["config"],
            ],
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        item["pid"] = process.pid
        save()
        result = process.wait()
    item["exit_code"] = result
    item["finished_at"] = datetime.now(timezone.utc).isoformat()
    item["status"] = "completed" if result == 0 else "failed"
    save()
    if result:
        state["status"] = "failed"
        save()
        print("TRAIN_FAILED", name, result, flush=True)
        raise SystemExit(result)
    print("TRAIN_COMPLETED", name, flush=True)
state["status"] = "completed"
save()
