import torch.nn.functional as F

from .base import Objective


class SoftCEObjective(Objective):
    needs_teacher = True

    def forward(self, prediction, supervision):
        if supervision.target_logits is None:
            raise ValueError("Soft objective requires frozen teacher logits")
        draft_log_probabilities = F.log_softmax(prediction.draft_logits.float(), -1)
        target_log_probabilities = F.log_softmax(
            supervision.target_logits.detach().float(), -1
        )
        token_losses = -(target_log_probabilities.exp() * draft_log_probabilities).sum(
            -1
        )
        if self.recipe["type"] == "forward_kl":
            token_losses = token_losses + (
                target_log_probabilities.exp() * target_log_probabilities
            ).sum(-1)
        loss_weights = self.loss_weights(supervision)
        return self.output((token_losses * loss_weights).sum(), loss_weights.sum())
