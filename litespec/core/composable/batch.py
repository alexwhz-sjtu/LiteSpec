"""Generic anchor/layout preparation, outside trainable math."""

import os
import tempfile
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Optional

import torch

from litespec.core.block_training import (
    apply_exponential_position_decay,
    gather_block_values,
    sample_sorted_anchors,
)


@dataclass
class BlockLayout:
    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor
    labels: torch.Tensor
    previous_token_ids: torch.Tensor
    prediction_mask: torch.Tensor
    label_indices: torch.Tensor
    sample_prediction_counts: torch.Tensor

    def to(self, device):
        return BlockLayout(
            **{f.name: getattr(self, f.name).to(device) for f in fields(self)}
        )

    def batch_slice(self, start, end):
        return BlockLayout(
            **{f.name: getattr(self, f.name)[start:end] for f in fields(self)}
        )


def prepare_layout(
    input_ids, loss_mask, block_size, num_anchors, *, complete_blocks=False
):
    length = input_ids.size(1)
    eligible = (loss_mask[:, :-1] > 0.5) & (loss_mask[:, 1:] > 0.5)
    if complete_blocks:
        offsets = torch.arange(block_size, device=input_ids.device)
        indices = (
            torch.arange(max(0, length - block_size + 1), device=input_ids.device)[
                :, None
            ]
            + offsets
        )
        whole = (loss_mask[:, indices] > 0.5).all(-1)
        eligible = torch.nn.functional.pad(
            whole, (0, max(0, length - 1 - whole.size(1)))
        )
    if eligible.numel() and eligible.any():
        num_anchors, keep = sample_sorted_anchors(
            eligible, num_anchors, output_width=num_anchors
        )
    else:
        num_anchors = input_ids.new_zeros(input_ids.size(0), num_anchors)
        keep = torch.zeros_like(num_anchors, dtype=torch.bool)
    labels, in_bounds, indices = gather_block_values(input_ids, num_anchors, block_size)
    mask, _, _ = gather_block_values(loss_mask, num_anchors, block_size)
    mask = mask.float() * keep[..., None] * in_bounds
    mask = mask[..., 1:]
    labels = labels[..., 1:]
    previous = torch.cat(
        (input_ids.gather(1, num_anchors)[..., None], labels[..., :-1]), -1
    )
    return BlockLayout(
        num_anchors,
        keep,
        labels,
        previous,
        mask,
        indices[..., 1:],
        mask.flatten(1).sum(-1),
    )


def position_weights(mask, gamma):
    return apply_exponential_position_decay(
        mask.float(),
        gamma,
        torch.arange(mask.size(-1), device=mask.device),
        first_prediction_index=0,
    )


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    if is_dataclass(value):
        return type(value)(
            **{f.name: cpu_copy(getattr(value, f.name)) for f in fields(value)}
        )
    return value


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    if is_dataclass(value):
        return sum(tensor_bytes(getattr(value, f.name)) for f in fields(value))
    return 0


@dataclass
class PreparedLogicalBatch:
    data: object
    layout: BlockLayout
    window_denominators: torch.Tensor
    spill_path: Optional[str] = None

    def materialize(self):
        if self.spill_path and self.data is None:
            # Locally produced trusted tensor packet, never a dataset-provided pickle.
            self.data = torch.load(
                self.spill_path, map_location="cpu", weights_only=False
            )
        return self.data

    def release(self):
        if self.spill_path:
            Path(self.spill_path).unlink(missing_ok=True)
            self.spill_path = None
        self.data = None

    def spill(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="window-", suffix=".pt", dir=directory)
        os.close(fd)
        torch.save(self.data, name)
        self.spill_path, self.data = name, None
