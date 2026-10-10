"""DFlash2 candidate-selector training objective."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._dflash_common import (
    SelectorTerms,
    _DFlashSelectorDiagnostics,
    _DFlashUnaryDiagnostics,
)
from .dflash import OnlineDFlashModel


class OnlineDFlash2Model(OnlineDFlashModel):
    def _selector_chunk_terms(
        self,
        candidate_selector: nn.Module,
        objective_logits: torch.Tensor,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
        predecessor_ids: torch.Tensor,
        loss_weights: torch.Tensor,
        weight_mask: torch.Tensor,
    ) -> SelectorTerms:
        """Return additive selector terms for one enabled objective chunk.

        The caller gates this helper only on the model-static selector
        configuration, never on the per-step effective selector alpha. During
        warmup the zero-scaled CE term must remain in the autograd graph so DDP
        with ``find_unused_parameters=False`` observes every selector parameter.
        The caller flattens these fields into ``DFlashObjectiveTerms`` to preserve
        ``checkpointed_chunk_reduce``'s flat tuple contract.
        """

        if self.selector_stop_gradient:
            # Isolate only the selector objective. The caller still uses the
            # original tensors for the primary DFlash objective.
            objective_logits = objective_logits.detach()
            hidden = hidden.detach()

        # Match serving exactly: train only against the strict unary top-k.
        # Candidate misses are a backbone/recall failure, not a selector
        # classification example, so they carry no selector gradient.
        unary_logits, candidate_ids = objective_logits.topk(
            candidate_selector.top_k,
            dim=-1,
        )
        target_matches = candidate_ids.eq(target_ids.unsqueeze(-1))
        target_is_candidate = target_matches.any(dim=-1)
        target_candidate_index = target_matches.long().argmax(dim=-1)
        selector_logits = candidate_selector.score_candidates(
            candidate_ids=candidate_ids,
            unary_logits=unary_logits,
            hidden_states=hidden,
            predecessor_ids=predecessor_ids,
        )
        selector_ce = F.cross_entropy(
            selector_logits.float().reshape(-1, selector_logits.shape[-1]),
            target_candidate_index.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)
        selector_probability = torch.exp(-selector_ce)
        selector_loss_weights = loss_weights * target_is_candidate.float()
        selector_metric_mask = weight_mask * target_is_candidate.float()
        ce_num = (selector_ce * selector_loss_weights).sum()
        probability_num = (selector_probability.detach() * selector_metric_mask).sum()
        weight_den = selector_loss_weights.sum()
        covered_num = selector_metric_mask.sum()
        with torch.no_grad():
            selected_ids = candidate_ids.gather(
                -1,
                selector_logits.argmax(dim=-1, keepdim=True),
            ).squeeze(-1)
            correct_num = (
                (selected_ids == target_ids).float() * selector_loss_weights
            ).sum()
        return SelectorTerms(
            ce_num=ce_num,
            probability_num=probability_num,
            correct_num=correct_num,
            weight_den=weight_den,
            covered_num=covered_num,
        )

    @staticmethod
    def _dflash2_selector_diagnostics(
        *,
        candidate_selector: nn.Module,
        unary: _DFlashUnaryDiagnostics,
        hidden: torch.Tensor,
        predecessor_ids: torch.Tensor,
        weight_mask: torch.Tensor,
    ) -> _DFlashSelectorDiagnostics:
        """Evaluate the selector with ground-truth predecessor tokens."""

        selector_logits = candidate_selector.score_candidates(
            candidate_ids=unary.candidate_ids,
            unary_logits=unary.topk_logits,
            hidden_states=hidden,
            predecessor_ids=predecessor_ids,
        )
        target_is_candidate = unary.target_is_candidate.float()
        return _DFlashSelectorDiagnostics(
            selected_ids=unary.candidate_ids.gather(
                -1,
                selector_logits.argmax(dim=-1, keepdim=True),
            ).squeeze(-1),
            covered_mask=weight_mask * target_is_candidate,
            logits=selector_logits,
        )
