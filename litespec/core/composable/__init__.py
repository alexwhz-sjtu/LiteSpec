"""Training-only wrapper and objectives for composable drafts."""

from .objectives import build_objective
from .training_model import ComposableTrainingModel

__all__ = ["ComposableTrainingModel", "build_objective"]
