"""Lucas-TY/D-PACE f36bad6e: detached credit, native / original batch."""

import torch
import torch.nn.functional as F

from .base import Objective


class DPACEObjective(Objective):
    sample_reduction = True

    def __init__(self, recipe, proposal_type="parallel"):
        super().__init__(recipe, proposal_type)
        self.alpha = float(recipe.get("alpha", 0.5))
        if not 0 <= self.alpha <= 1:
            raise ValueError("D-PACE alpha must be in [0,1]")

    def forward(self, prediction, supervision):
        draft_logits = prediction.draft_logits
        token_losses = F.cross_entropy(
            draft_logits.flatten(0, -2), supervision.labels.flatten(), reduction="none"
        ).reshape_as(supervision.prediction_mask)
        mask = supervision.prediction_mask.float()
        with torch.no_grad():
            draft_label_probabilities = torch.exp(-token_losses)
            smooth = (1 - self.alpha) * draft_label_probabilities + self.alpha
            smooth = torch.where(mask > 0, smooth, torch.ones_like(smooth))
            prefix = torch.cumprod(smooth, -1)
            credit = torch.flip(torch.cumsum(torch.flip(prefix * mask, [-1]), -1), [-1])
        numerator = (token_losses * mask * credit).sum()
        return self.output(numerator, numerator.new_tensor(draft_logits.size(0)))
