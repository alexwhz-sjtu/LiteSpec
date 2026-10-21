"""Run the real training lifecycle for two steps without changing production defaults."""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litespec.training.runner import configure_rank_caches

configure_rank_caches()
from litespec.training.algorithms import get_training_adapter
from litespec.training.runner import run_training

algorithm = sys.argv.pop(1)
adapter = get_training_adapter(algorithm)
args = adapter.parse_args(argv=sys.argv[1:])
original_phases = adapter.phases
adapter.phases = lambda values: [
    dataclasses.replace(p, epochs=1, max_batches=2) for p in original_phases(values)
]
args.final_save = False
run_training(args, adapter)
