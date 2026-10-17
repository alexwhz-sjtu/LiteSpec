"""Confirmed-context GDN state, separate from attention and proposal caches."""

from dataclasses import dataclass, field

import torch


@dataclass
class GDNLayerState:
    recurrent_state: torch.Tensor
    kv_conv_state: torch.Tensor | None
    context_conv_kernel_size: int


@dataclass
class GDNBackboneState:
    layers: dict[int, GDNLayerState] = field(default_factory=dict)
    next_positions: torch.Tensor | None = None
    document_ids: torch.Tensor | None = None

    def validate_rollback(self, next_position):
        if self.next_positions is None:
            return
        limits = torch.as_tensor(next_position, device=self.next_positions.device)
        if limits.ndim > 1:
            limits = limits.squeeze(-1)
        if (limits < self.next_positions).any():
            raise ValueError(
                "GDN cannot crop confirmed recurrent history; restore a complete "
                "state snapshot or rebuild the requested prefix"
            )
