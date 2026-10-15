# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: public_reference
# Repository: vllm-project/speculators
# Revision: c5d4f9dec9bddcad48b5611344ba19b8b6039dc0
# Source: src/speculators/losses/utils.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/vllm-project__speculators/c5d4f9dec9bddcad48b5611344ba19b8b6039dc0/src/speculators/losses/utils.py
# License / upstream notices: refrence/licenses/nla-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Preserve per-sample valid-mask denominator then sample mean, including epsilon; decay enters numerator only.


# BEGIN _LOSS_REDUCTION_EPS — upstream lines 10-10
_LOSS_REDUCTION_EPS = 1e-5

# END _LOSS_REDUCTION_EPS

# BEGIN dflash_loss_decay — upstream lines 17-48
def dflash_loss_decay(
    pos_idx: torch.Tensor, gamma: float, sample_from_anchor: bool = False, **_kwargs
):
    """Compute DFlash-style exponential decay weights per position.

    When ``sample_from_anchor`` is False (DFlash), position 0 is the anchor and
    gets weight 0; position 1 gets weight 1, and subsequent positions decay as
    ``exp(-(pos - 1) / gamma)``.

    When ``sample_from_anchor`` is True (DSpark), position 0 is the first real
    prediction and gets weight 1; subsequent positions decay as
    ``exp(-pos / gamma)``.

    Args:
        pos_idx: Position indices within each speculative block.
        gamma: Decay rate (higher = slower decay).
        sample_from_anchor: Whether position 0 is a prediction (DSpark) or the
            anchor input (DFlash).

    Returns:
        Decay multiplier tensor with same shape as pos_idx.
    """
    if sample_from_anchor:
        # DSpark: slot 0 predicts the token after the anchor, so decay from 0.
        return torch.exp(-pos_idx / gamma)
    # DFlash: slot 0 is the anchor (no prediction); decay from slot 1.
    # pos_idx = 0 1 2 3 0 1 2 3, block_size = 4
    decay_mult = torch.exp(-((pos_idx - 1).clamp(min=0)) / gamma)
    # decay_mult = e^-(0 0 1 2 0 0 1 2) / gamma
    decay_mult = decay_mult * (pos_idx != 0).to(decay_mult.dtype)
    # w = 0 1 e^-1/gamma e^-2/gamma 0 1 e^-1/gamma e^-2/gamma
    return decay_mult  # noqa: RET504

# END dflash_loss_decay

# BEGIN loss_function — upstream lines 269-304
def loss_function(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    loss_mask: torch.Tensor,  # shape: [1, seq_len]
    pos_idx: torch.Tensor,  # shape: [1, seq_len]
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = kl_div_loss,
    decay_fn: Callable[..., torch.Tensor] | None = None,
):
    """Compute masked, optionally position-decayed training loss.

    Args:
        logits: Draft model logits.
        targets: Target model logits.
        loss_mask: Boolean mask selecting positions to include in the loss.
        pos_idx: Position indices within each speculative block.
        loss_fn: Per-position loss function (default: kl_div_loss).
        decay_fn: Optional position-dependent decay weighting function.

    Returns:
        Scalar mean loss across the batch.
    """
    elementwise_loss = loss_fn(logits, targets)  # shape: [1, seq_len]

    loss_mask = loss_mask.to(elementwise_loss.dtype)
    elementwise_loss = elementwise_loss * loss_mask

    if decay_fn is not None:
        decay_mult = decay_fn(
            pos_idx.to(elementwise_loss.dtype), elementwise_loss=elementwise_loss
        )
        elementwise_loss = elementwise_loss * decay_mult

    denominator = loss_mask.sum(dim=1) + _LOSS_REDUCTION_EPS

    batch_loss = torch.sum(elementwise_loss, dim=1) / denominator  # shape: [1]
    return batch_loss.mean()  # shape: []

# END loss_function
