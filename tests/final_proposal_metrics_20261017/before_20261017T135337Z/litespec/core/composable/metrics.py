"""Logging-only acceptance diagnostics using the shared block definitions."""

import torch

from litespec.core.block_metrics import (
    block_accepted_length_terms,
    distribution_acceptance_probability,
    expected_accept_len_terms,
    sparse_distribution_acceptance_probability,
)


@torch.no_grad()
def acceptance_metric_terms(
    proposal, prediction, target_logits, labels, previous_token_ids, prediction_mask
):
    """Return additive walk/expected numerators and the valid-block count.

    Walks use free-running greedy predecessors, starting at each ground-truth
    anchor. Expected lengths use teacher-forced conditional proposal rows, just
    like DFlash2's training diagnostic; they are not a rollout evaluation.
    Both lengths include the anchor and stop at the first unsupervised slot.
    """
    supervised = prediction_mask > 0.5
    hidden = prediction.draft_hidden_states
    anchor_ids = previous_token_ids[..., 0].reshape(-1)
    shape = labels.shape
    if hasattr(proposal, "candidate_selector"):
        selector = proposal.candidate_selector
        unary = prediction.base_logits.gather(-1, prediction.candidate_ids)
        greedy_ids = selector.greedy_path(
            candidate_ids=prediction.candidate_ids.flatten(0, 1),
            unary_logits=unary.flatten(0, 1),
            hidden_states=hidden.flatten(0, 1),
            anchor_token_ids=anchor_ids,
        ).reshape(shape)
        acceptance = sparse_distribution_acceptance_probability(
            target_logits, prediction.candidate_logits, prediction.candidate_ids
        )
    else:
        if hasattr(proposal, "markov_head"):
            greedy_ids, _ = proposal.markov_head.sample_block_tokens(
                prediction.base_logits.flatten(0, 1),
                hidden_states=hidden.flatten(0, 1),
                first_prev_token_ids=anchor_ids,
                temperature=0.0,
            )
            greedy_ids = greedy_ids.reshape(shape)
        elif hasattr(proposal, "sequential_head"):
            greedy_ids, _ = proposal.sequential_head.sample_block_tokens(
                hidden_states=hidden.flatten(0, 1),
                first_prev_token_ids=anchor_ids,
                temperature=0.0,
            )
            greedy_ids = greedy_ids.reshape(shape)
        else:
            greedy_ids = prediction.draft_logits.argmax(-1)
        acceptance = distribution_acceptance_probability(
            target_logits, prediction.draft_logits
        )
    walk = block_accepted_length_terms(greedy_ids, labels, supervised)
    expected = expected_accept_len_terms(acceptance, supervised)
    return walk.numerator, expected.numerator, walk.denominator
