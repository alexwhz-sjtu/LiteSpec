import torch

from .base import Objective


class TVObjective(Objective):
    needs_teacher = True

    def forward(self, prediction, supervision):
        draft_probabilities = torch.softmax(prediction.draft_logits.float(), -1)
        target_probabilities = torch.softmax(
            supervision.target_logits.detach().float(), -1
        )
        token_losses = 0.5 * (target_probabilities - draft_probabilities).abs().sum(-1)
        loss_weights = self.loss_weights(supervision)
        return self.output((token_losses * loss_weights).sum(), loss_weights.sum())
