"""Audit real optimizer groups while running the existing two-update smoke."""
import json
import os
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litespec.training.runner import configure_rank_caches
configure_rank_caches()
from litespec.training.optimizer import BF16Optimizer
original_init = BF16Optimizer.__init__
def audited_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    groups = {name: [float(group["weight_decay"]) for group in optimizer.param_groups]
              for name, optimizer in (("adamw", self.optimizer), ("muon", self.muon_optimizer))
              if optimizer is not None}
    assert groups and all(value == 0.0 for values in groups.values() for value in values), groups
    rank = os.environ.get("RANK", "0")
    path = Path(__file__).with_name(f"wd0_optimizer_groups_rank{rank}.json")
    path.write_text(json.dumps(groups, indent=2) + "\n")
    print("WD0_OPTIMIZER_VERIFIED", rank, json.dumps(groups), flush=True)
BF16Optimizer.__init__ = audited_init
# The original script also configures rank caches; restore the first entrypoint
# environment so that those paths are not nested twice.
for key in ("TVM_FFI_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
    os.environ.pop(key, None)
runpy.run_path(str(Path(__file__).with_name("realsemi_training_smoke.py")), run_name="__main__")
