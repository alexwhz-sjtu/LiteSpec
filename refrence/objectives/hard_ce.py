# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: existing_framework
# Repository: alexwhz-sjtu/LiteSpec
# Revision: 7fa8b08ddaa29d619dc193d073185e197f77b230
# Source: litespec/core/losses.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/alexwhz-sjtu__LiteSpec/7fa8b08ddaa29d619dc193d073185e197f77b230/litespec/core/losses.py
# License / upstream notices: refrence/licenses/litespec-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Hard labels and numerator only; outer reduction is a separate contract.


# BEGIN _ce — upstream lines 19-23
def _ce(logits, labels, weights):
    values = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).reshape_as(labels)
    return (values * weights).sum(), values.detach()

# END _ce

# BEGIN weighted_cross_entropy — upstream lines 109-115
def weighted_cross_entropy(logits, labels, weights):
    """Return an additive numerator and detached, unweighted per-token CE."""
    # FP32 logits may have been cast from a BF16 projection. Compiling CE's
    # log-softmax can shift its FP32 gradient across BF16 rounding midpoints,
    # changing model gradients and optimizer updates despite a close loss.
    _should_compile(logits)  # Validate the configured backend on this path.
    return _ce(logits, labels, weights)

# END weighted_cross_entropy
