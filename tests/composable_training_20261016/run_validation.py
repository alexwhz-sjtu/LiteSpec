import json
import os
import re
import subprocess
import time
from pathlib import Path

project = Path(__file__).resolve().parents[2]
root = Path(__file__).resolve().parent
config_path = project / "scripts/config/composable.yaml"
original = (root / "composable.before.yaml").read_text()
status_path = root / "status.json"
results = []

def status(**extra):
    status_path.write_text(json.dumps({"results": results, **extra}, indent=2))

status(state="starting", started=time.time())
try:
    for name, draft_path in [
        ("default", "configs/composable_draft.json"),
        ("heterogeneous", str(project / "configs/composable_heterogeneous.json")),
    ]:
        configured = re.sub(r"(?m)^  draft_config:.*$", "  draft_config: " + draft_path, original)
        configured = re.sub(r"(?m)^  output_dir:.*$", "  output_dir: " + str(root / ("output_" + name)), configured)
        configured += "  max_steps: 500\n"
        config_path.write_text(configured)
        (root / (name + ".yaml")).write_text(configured)
        env = dict(os.environ, WANDB_MODE="offline", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
        log_path = root / (name + ".log")
        with log_path.open("w") as stream:
            proc = subprocess.Popen(
                ["bash", "scripts/composable/run_training_composable.sh", "scripts/config/composable.yaml"],
                cwd=project, env=env, stdout=stream, stderr=subprocess.STDOUT,
            )
            status(state="running", case=name, pid=proc.pid, started=time.time())
            returncode = proc.wait()
        log = log_path.read_text(errors="replace")
        metrics = re.findall(r"composable optimizer_step=(\d+) micro_step=(\d+) loss=([^\s]+)", log)
        reached = max([int(step) for step, _, _ in metrics], default=0)
        result = {"case": name, "returncode": returncode, "optimizer_steps": reached, "metrics": metrics, "log": str(log_path)}
        results.append(result)
        status(state="case_completed", case=name)
        if returncode != 0 or reached != 500:
            raise RuntimeError("Validation failed: " + json.dumps(result))
    status(state="complete", finished=time.time())
finally:
    final_config = original
    if results and results[0]["returncode"] == 0 and results[0]["optimizer_steps"] == 500:
        final_config = re.sub(
            r"(?m)^  draft_config:.*$",
            "  draft_config: " + str(project / "configs/composable_heterogeneous.json"),
            original,
        )
    config_path.write_text(final_config)
