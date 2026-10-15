# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: public_reference
# Repository: vllm-project/speculators
# Revision: c5d4f9dec9bddcad48b5611344ba19b8b6039dc0
# Source: src/speculators/losses/eager.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/vllm-project__speculators/c5d4f9dec9bddcad48b5611344ba19b8b6039dc0/src/speculators/losses/eager.py
# License / upstream notices: refrence/licenses/nla-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Standard TV; do not confuse with framework L1=2*TV.


# BEGIN tv_loss — upstream lines 118-142
def tv_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Compute per-position total variation (TV) distance from draft to target.

    The rejection-sampling acceptance rate of speculative decoding equals the
    distributional overlap between target and draft,
    ``alpha = sum_v min(p_v, q_v) = 1 - d_TV(p, q)``. Minimizing this TV distance
    therefore directly optimizes the acceptance rate, whereas cross-entropy and
    KL only optimize it indirectly (KL is a loose upper bound on TV via Pinsker).

    Args:
        logits: Draft model logits (softmax applied internally to form q).
        targets: Target model logits (softmax applied internally to form p).

    Returns:
        Per-position TV distance with shape [1, seq_len].
    """
    draft_p = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
    target_p = torch.nn.functional.softmax(targets, dim=-1, dtype=torch.float32)
    overlap = torch.minimum(draft_p, target_p).sum(dim=-1)  # shape: [1, seq_len]
    elementwise_loss = 1.0 - overlap

    return elementwise_loss  # noqa: RET504

# END tv_loss
