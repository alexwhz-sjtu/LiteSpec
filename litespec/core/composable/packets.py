"""Full-hidden transport with explicit context padding validity."""

from dataclasses import dataclass

import torch

from litespec.distributed.transport import (
    DFlashFamilyBatchPacket,
    DFlashFamilyPacketSpec,
)


@dataclass(frozen=True)
class ComposablePacketSpec(DFlashFamilyPacketSpec):
    @property
    def schema_id(self):
        return 30 + int(self.include_target_last_hidden)


@dataclass
class ComposableBatchPacket(DFlashFamilyBatchPacket):
    context_valid_mask: torch.Tensor

    @classmethod
    def empty(cls, spec, *, device="cuda"):
        packet = DFlashFamilyBatchPacket.empty(spec, device=device)
        return cls(
            packet.input_ids,
            packet.loss_mask,
            packet.target_context_hidden,
            packet.target_last_hidden,
            torch.empty(
                spec.batch_size, spec.max_length, dtype=torch.bool, device=device
            ),
        )
