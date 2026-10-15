# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: framework_integration_baseline
# Repository: alexwhz-sjtu/LiteSpec
# Revision: 7fa8b08ddaa29d619dc193d073185e197f77b230
# Source: litespec/training/optimizer.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/alexwhz-sjtu__LiteSpec/7fa8b08ddaa29d619dc193d073185e197f77b230/litespec/training/optimizer.py
# License / upstream notices: refrence/licenses/litespec-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Do not pre-divide D-PACE numerator and then let this function divide by N again.


# BEGIN normalize_accumulated_gradients — upstream lines 732-759
def normalize_accumulated_gradients(
    optimizer,
    local_denominator: torch.Tensor,
    accumulation_steps: int,
    group=None,
) -> float:
    """Normalize rank-averaged gradients by the global optimizer-window weight.

    Every microbatch backpropagates its additive loss numerator divided by the
    configured accumulation length. FSDP averages those gradients across ranks.
    Multiplying by ``world_size * accumulation_steps / global_denominator``
    therefore produces the gradient of the summed numerator divided by the sum
    of effective weights across every microbatch and DP rank in the window.
    """
    if int(accumulation_steps) <= 0:
        raise ValueError("accumulation_steps must be positive")
    denominator = local_denominator.detach().float().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size(group=group)
        if world_size > 1:
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=group)
    denominator_value = float(denominator.item())
    if not math.isfinite(denominator_value) or denominator_value <= 0:
        raise ValueError("global loss denominator must be finite and positive")
    scale = world_size * int(accumulation_steps) / denominator_value
    optimizer.scale_model_gradients(scale)
    return denominator_value

# END normalize_accumulated_gradients
