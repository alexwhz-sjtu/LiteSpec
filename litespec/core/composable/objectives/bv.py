"""BV paper reproduction, arXiv:2609.34832v1; no located author source.
Integrated scores keep path-prefix gradients. Annealing floor depends ONLY on
whether the proposal is plain parallel (1e-6) or has a correction head (none).
"""

import math

import torch

from .base import Objective


def integrated_log_scores(draft_log_probabilities, target_log_probabilities, labels):
    log_ratio = torch.zeros_like(labels[..., 0], dtype=torch.float32)
    log_acceptance = torch.zeros_like(log_ratio)
    scores = []
    for position in range(labels.size(-1)):
        draft_step_log_probabilities = draft_log_probabilities[..., position, :]
        target_step_log_probabilities = target_log_probabilities[..., position, :]
        scores.append(
            torch.logsumexp(
                torch.minimum(
                    log_acceptance[..., None] + target_step_log_probabilities,
                    log_ratio[..., None] + draft_step_log_probabilities,
                ),
                -1,
            )
        )
        ids = labels[..., position, None]
        log_ratio = (
            log_ratio
            + draft_step_log_probabilities.gather(-1, ids).squeeze(-1)
            - target_step_log_probabilities.gather(-1, ids).squeeze(-1)
        )
        log_acceptance = torch.minimum(log_acceptance, log_ratio)
    return torch.stack(scores, -1)


class BVObjective(Objective):
    needs_teacher = True

    def __init__(self, recipe, proposal_type="parallel"):
        super().__init__(recipe, proposal_type)
        if proposal_type == "selector":
            raise ValueError("BV + selector is unsupported")
        self.annealing_score_floor = 1e-6 if proposal_type == "parallel" else None

    def native_denominator(self, layout):
        return layout.prediction_mask.bool().all(-1).sum().float()

    def forward(self, prediction, supervision):
        draft_log_probabilities = torch.log_softmax(prediction.draft_logits.float(), -1)
        target_log_probabilities = torch.log_softmax(
            supervision.target_logits.detach().float(), -1
        )
        log_scores = integrated_log_scores(
            draft_log_probabilities, target_log_probabilities, supervision.labels
        )
        complete = supervision.prediction_mask.bool().all(-1)
        beta = min(max(float(supervision.epoch_progress) / 3.0, 0.0), 1.0)
        if beta < 1:
            if self.annealing_score_floor is not None:
                log_scores = log_scores.clamp_min(math.log(self.annealing_score_floor))
            weights = torch.softmax(beta * log_scores, -1).detach()
            training_scalar = -(weights * log_scores).sum(-1)
            mathematical_loss = (
                -log_scores.mean(-1)
                if beta == 0
                else -(
                    torch.logsumexp(beta * log_scores, -1)
                    - math.log(log_scores.size(-1))
                )
                / beta
            )
        else:
            training_scalar = (
                -torch.exp(log_scores.clamp_min(-80)).sum(-1).clamp_min(1e-6).log()
            )
            mathematical_loss = training_scalar
        numerator = (training_scalar * complete).sum()
        return self.output(
            numerator,
            complete.sum().to(numerator),
            **{"bv/mathematical_loss": (mathematical_loss * complete).sum()},
        )
