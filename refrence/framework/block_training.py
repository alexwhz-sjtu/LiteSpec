# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: existing_framework
# Repository: alexwhz-sjtu/LiteSpec
# Revision: 7fa8b08ddaa29d619dc193d073185e197f77b230
# Source: litespec/core/block_training.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/alexwhz-sjtu__LiteSpec/7fa8b08ddaa29d619dc193d073185e197f77b230/litespec/core/block_training.py
# License / upstream notices: refrence/licenses/litespec-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Reuse these operations, with per-objective eligibility and untouched logical N/A/B.


# BEGIN sample_sorted_anchors — upstream lines 11-62
def sample_sorted_anchors(
    valid_anchor_mask: torch.Tensor,
    num_anchors: int,
    *,
    output_width: Optional[int] = None,
    max_valid_anchors: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly sample valid anchors, sort them, and pad with inactive slots.

    Returns ``anchor_positions`` and ``block_keep_mask``. ``output_width`` only
    sets the physical padded width; it does not change which sequence positions
    are eligible or how many real anchors are sampled.
    """

    if valid_anchor_mask.ndim != 2:
        raise ValueError("valid_anchor_mask must have shape (batch, candidates)")
    valid_anchor_mask = valid_anchor_mask.bool()
    counts = valid_anchor_mask.sum(dim=1)
    if max_valid_anchors is None:
        max_valid_anchors = int(counts.max().item())
    sampled_width = min(int(num_anchors), max(0, int(max_valid_anchors)))
    if sampled_width <= 0:
        raise ValueError("No valid anchor positions in this batch.")
    width = sampled_width if output_width is None else int(output_width)
    if width < sampled_width:
        raise ValueError("output_width cannot be smaller than sampled width")

    scores = torch.rand_like(valid_anchor_mask, dtype=torch.float32)
    scores.masked_fill_(~valid_anchor_mask, 2.0)
    selected_width = min(sampled_width, valid_anchor_mask.size(1))
    anchor_positions = scores.argsort(dim=1)[:, :selected_width]
    selected_valid = torch.gather(valid_anchor_mask, 1, anchor_positions)
    sentinel = valid_anchor_mask.size(1)
    anchor_positions = anchor_positions.masked_fill(~selected_valid, sentinel)
    if selected_width < width:
        anchor_positions = torch.cat(
            [
                anchor_positions,
                anchor_positions.new_full(
                    (anchor_positions.size(0), width - selected_width), sentinel
                ),
            ],
            dim=1,
        )
    anchor_positions = anchor_positions.sort(dim=1).values
    block_keep_mask = anchor_positions != sentinel
    return (
        torch.where(
            block_keep_mask, anchor_positions, torch.zeros_like(anchor_positions)
        ),
        block_keep_mask,
    )

# END sample_sorted_anchors

# BEGIN gather_block_values — upstream lines 65-89
def gather_block_values(
    values: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather ``anchor + offset`` values and return values, bounds, safe indices."""

    if values.ndim < 2 or anchor_positions.ndim != 2:
        raise ValueError(
            "values and anchor_positions must have batch and sequence axes"
        )
    offsets = torch.arange(int(block_size), device=anchor_positions.device).view(
        1, 1, -1
    )
    indices = anchor_positions.unsqueeze(-1) + offsets
    in_bounds = indices < values.size(1)
    safe_indices = indices.clamp(min=0, max=max(values.size(1) - 1, 0))
    expanded = values.unsqueeze(1).expand(
        -1, anchor_positions.size(1), *values.shape[1:]
    )
    gather_index = safe_indices
    for _ in values.shape[2:]:
        gather_index = gather_index.unsqueeze(-1)
    gather_index = gather_index.expand(-1, -1, -1, *values.shape[2:])
    return torch.gather(expanded, 2, gather_index), in_bounds, safe_indices

# END gather_block_values

# BEGIN apply_exponential_position_decay — upstream lines 92-106
def apply_exponential_position_decay(
    raw_weights: torch.Tensor,
    gamma: Optional[float],
    position_indices: torch.Tensor,
    *,
    first_prediction_index: int = 1,
) -> torch.Tensor:
    """Apply decay using semantic block positions rather than tensor offsets."""

    if gamma is None or gamma <= 0:
        return raw_weights
    positions = position_indices.to(device=raw_weights.device, dtype=torch.float32)
    decay_steps = (positions - int(first_prediction_index)).clamp_min(0)
    shape = (1,) * (raw_weights.ndim - 1) + (-1,)
    return raw_weights * torch.exp(-decay_steps / float(gamma)).view(shape)

# END apply_exponential_position_decay

# BEGIN aligned_width — upstream lines 109-115
def aligned_width(width: int, alignment: int) -> int:
    """Round a positive logical width to a backend-required physical width."""

    width, alignment = int(width), int(alignment)
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return math.ceil(width / alignment) * alignment

# END aligned_width
