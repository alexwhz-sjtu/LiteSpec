"""Shared tensor-only preparation helpers for block-parallel training."""

from __future__ import annotations

import math
from typing import Optional

import torch


def sample_sorted_anchors(
    valid_anchor_mask: torch.Tensor,
    max_anchors: int,
    *,
    output_width: Optional[int] = None,
    max_valid_anchors: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly sample valid anchors, sort them, and append invalid padding.

    ``output_width`` is deliberately separate from sampling.  Callers may use
    it for a backend-specific padded shape without changing which anchors are
    eligible or how many real anchors are sampled.
    """

    if valid_anchor_mask.ndim != 2:
        raise ValueError("valid_anchor_mask must have shape (batch, candidates)")
    valid = valid_anchor_mask.bool()
    counts = valid.sum(dim=1)
    if max_valid_anchors is None:
        max_valid_anchors = int(counts.max().item())
    real_width = min(int(max_anchors), max(0, int(max_valid_anchors)))
    if real_width <= 0:
        raise ValueError("No valid anchor positions in this batch.")
    width = real_width if output_width is None else int(output_width)
    if width < real_width:
        raise ValueError("output_width cannot be smaller than sampled width")

    scores = torch.rand_like(valid, dtype=torch.float32)
    scores.masked_fill_(~valid, 2.0)
    selected_width = min(real_width, valid.size(1))
    indices = scores.argsort(dim=1)[:, :selected_width]
    selected_valid = torch.gather(valid, 1, indices)
    sentinel = valid.size(1)
    indices = indices.masked_fill(~selected_valid, sentinel)
    if selected_width < width:
        indices = torch.cat(
            [
                indices,
                indices.new_full((indices.size(0), width - selected_width), sentinel),
            ],
            dim=1,
        )
    indices = indices.sort(dim=1).values
    keep = indices != sentinel
    return torch.where(keep, indices, torch.zeros_like(indices)), keep


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
    safe = indices.clamp(min=0, max=max(values.size(1) - 1, 0))
    expanded = values.unsqueeze(1).expand(
        -1, anchor_positions.size(1), *values.shape[1:]
    )
    gather_index = safe
    for _ in values.shape[2:]:
        gather_index = gather_index.unsqueeze(-1)
    gather_index = gather_index.expand(-1, -1, -1, *values.shape[2:])
    return torch.gather(expanded, 2, gather_index), in_bounds, safe


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


def aligned_width(width: int, alignment: int) -> int:
    """Round a positive logical width to a backend-required physical width."""

    width, alignment = int(width), int(alignment)
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return math.ceil(width / alignment) * alignment


__all__ = [
    "aligned_width",
    "apply_exponential_position_decay",
    "gather_block_values",
    "sample_sorted_anchors",
]
