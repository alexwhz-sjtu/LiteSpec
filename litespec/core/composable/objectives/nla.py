"""Speculators c5d4f9d public reference, not LK author-source parity."""

import torch

from .base import Objective


class NLAObjective(Objective):
    needs_teacher = True
    sample_reduction = True

    def forward(self, prediction, supervision):
        draft_probabilities = torch.softmax(prediction.draft_logits.float(), -1)
        target_probabilities = torch.softmax(
            supervision.target_logits.detach().float(), -1
        )
        token_losses = (
            -torch.minimum(target_probabilities, draft_probabilities)
            .sum(-1)
            .clamp_min(1e-5)
            .log()
        )
        numerator = (token_losses * self.loss_weights(supervision)).flatten(1).sum(-1)
        counts = supervision.sample_prediction_counts
        if counts is None:
            counts = supervision.prediction_mask.float().flatten(1).sum(-1)
        numerator = (numerator / (counts.float() + 1e-5)).sum()
        return self.output(
            numerator, numerator.new_tensor(prediction.draft_logits.size(0))
        )
