import torch.nn.functional as F

from .base import Objective


class HardCEObjective(Objective):
    def forward(self, prediction, supervision):
        draft_logits = prediction.draft_logits.float()
        token_losses = F.cross_entropy(
            draft_logits.flatten(0, -2), supervision.labels.flatten(), reduction="none"
        ).reshape_as(supervision.prediction_mask)
        loss_weights = self.loss_weights(supervision)
        return self.output((token_losses * loss_weights).sum(), loss_weights.sum())
