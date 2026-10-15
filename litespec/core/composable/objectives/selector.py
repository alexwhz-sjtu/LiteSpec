"""Parameter-free DFlash2 auxiliary CE; never registered as a main objective."""

import torch.nn.functional as F

from .base import Objective


class SelectorAuxiliaryObjective(Objective):
    def forward(self, prediction, supervision):
        covered = prediction.candidate_ids == supervision.labels[..., None]
        candidate_indices = covered.long().argmax(-1)
        token_losses = F.cross_entropy(
            prediction.candidate_logits.float().flatten(0, -2),
            candidate_indices.flatten(),
            reduction="none",
        ).reshape_as(supervision.prediction_mask)
        loss_weights = self.loss_weights(supervision)
        numerator = (token_losses * covered.any(-1) * loss_weights).sum()
        return self.output(numerator, loss_weights.sum())


def build_selector_objective(recipe):
    gamma = recipe["proposal"].get(
        "loss_decay_gamma", recipe["objective"].get("loss_decay_gamma")
    )
    return SelectorAuxiliaryObjective(
        {"type": "selector_auxiliary", "loss_decay_gamma": gamma}
    )
