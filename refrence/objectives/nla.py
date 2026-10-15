# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: public_reference
# Repository: vllm-project/speculators
# Revision: c5d4f9dec9bddcad48b5611344ba19b8b6039dc0
# Source: src/speculators/losses/eager.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/vllm-project__speculators/c5d4f9dec9bddcad48b5611344ba19b8b6039dc0/src/speculators/losses/eager.py
# License / upstream notices: refrence/licenses/nla-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Public Speculators reference, not an assertion that this is the LK paper authors code.


# BEGIN _NLA_EPS — upstream lines 8-8
_NLA_EPS = 1e-5

# END _NLA_EPS

# BEGIN neg_log_acceptance_loss — upstream lines 145-171
def neg_log_acceptance_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Compute per-position negative log-acceptance (LK) loss.

    The speculative-decoding acceptance rate equals the draft/target distribution
    overlap, ``alpha = sum_v min(p_v, q_v)`` (the same quantity computed in
    ``tv_loss``). This loss is ``-log(alpha)``. Its gradient is
    ``(1 / alpha) * grad(TV)``: the ``1 / alpha`` factor amplifies the otherwise
    vanishing TV gradient when overlap is low (early training), giving TV's
    acceptance-optimal target a usable gradient from a cold start. When the target
    is a point mass, this loss reduces to cross-entropy.

    Args:
        logits: Draft model logits (softmax applied internally to form q).
        targets: Target model logits (softmax applied internally to form p).

    Returns:
        Per-position negative log-acceptance with shape [1, seq_len].
    """
    draft_p = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
    target_p = torch.nn.functional.softmax(targets, dim=-1, dtype=torch.float32)
    overlap = torch.minimum(draft_p, target_p).sum(dim=-1)  # alpha, shape: [1, seq_len]
    elementwise_loss = -torch.log(overlap.clamp_min(_NLA_EPS))

    return elementwise_loss  # noqa: RET504

# END neg_log_acceptance_loss
