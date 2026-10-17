import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

project=Path(__file__).resolve().parents[2]
root=Path(__file__).resolve().parent
status_path=root/"status.json"
if status_path.exists():
    previous=json.loads(status_path.read_text())
    if previous.get("state") in ("starting", "running", "complete"):
        raise SystemExit("An existing run is already registered; refusing duplicate launch.")
started=time.time()
env=dict(os.environ, WANDB_MODE="online", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
for key in ("DRY_RUN","PET_NPROC_PER_NODE","PET_NNODES","PET_NODE_RANK","PET_MASTER_ADDR","PET_MASTER_PORT"):
    env.pop(key,None)
state={"state":"starting","started_at":datetime.now(timezone.utc).isoformat(),
       "command":["bash","scripts/composable/run_training_composable.sh","scripts/config/composable.yaml"],
       "configured_epochs":6,"max_steps":None,"wandb_mode":"online","supervisor_pid":os.getpid()}
def save():
    temporary=status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state,indent=2))
    temporary.replace(status_path)
save()
with (root/"training.log").open("w") as stream:
    proc=subprocess.Popen(state["command"],cwd=project,env=env,
                          stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    state.update(state="running",pid=proc.pid)
    (root/"training.pid").write_text(str(proc.pid))
    save()
    result=proc.wait()
log=(root/"training.log").read_text(errors="replace")
metrics=re.findall(r"composable optimizer_step=(\d+) micro_step=(\d+) loss=([^\s]+)",log)
outputs=re.findall(r"Training output directory: ([^\r\n]+)",log)
links=re.findall(r"https://(?:wandb.ai|forge.coreweave.com)/[^\s\x1b]+/runs/[A-Za-z0-9]+",log)
state.update(state="complete" if result==0 else "failed",returncode=result,
             finished_at=datetime.now(timezone.utc).isoformat(),
             elapsed_seconds=time.time()-started,last_metrics=metrics[-1:] or None,
             output_dir=outputs[-1] if outputs else None,wandb_url=links[-1] if links else None)
save()
