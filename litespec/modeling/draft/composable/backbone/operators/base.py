"""Structural interface implemented by the source-matched local operator."""

from typing import Protocol

import torch


class LocalOperator(Protocol):
    def prepare(
        self, draft_hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def finish(
        self, draft_hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor: ...
