"""Parameter-free native objective terms; reduction belongs to the runner."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from litespec.core.outputs import RatioTerms, TrainingOutput

from ..batch import position_weights


@dataclass
class Supervision:
    labels: torch.Tensor
    prediction_mask: torch.Tensor
    target_logits: Optional[torch.Tensor] = None
    sample_prediction_counts: Optional[torch.Tensor] = None
    epoch_progress: float = 0.0


class Objective(nn.Module):
    needs_teacher = False
    sample_reduction = False

    def __init__(self, recipe, proposal_type="parallel"):
        super().__init__()
        self.recipe = dict(recipe)
        self.proposal_type = proposal_type

    def loss_weights(self, supervision):
        return position_weights(
            supervision.prediction_mask, self.recipe.get("loss_decay_gamma")
        )

    def native_denominator(self, layout):
        """Detached layout-only count for independent optimizer-window reductions."""
        if self.sample_reduction:
            return layout.prediction_mask.new_tensor(layout.prediction_mask.size(0))
        return self.loss_weights(layout).sum()

    def output(self, numerator, denominator, **metrics):
        terms = RatioTerms(numerator, denominator.detach().to(numerator))
        ratio_metrics = {
            name: RatioTerms(value.detach(), terms.denominator)
            for name, value in metrics.items()
        }
        return TrainingOutput(
            numerator / denominator.clamp_min(1e-12), terms, ratio_metrics
        )
