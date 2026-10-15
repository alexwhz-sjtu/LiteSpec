"""Reuse LiteSpec Qwen projections; DFlare gets dedicated context K/V."""

from copy import deepcopy

import torch

from ....dflash import Qwen3DFlashAttention, apply_rotary_pos_emb
from .access import build_access
from .base import AttentionForward


class MHAAttention(AttentionForward, Qwen3DFlashAttention):
    def __init__(self, config, layer_idx, kernels, recipe, separate_context=False):
        super().__init__(config, layer_idx, kernels)
        self.backend = recipe.get("backend", "sdpa")
        self.flex_backend = recipe.get("flex_backend", "TRITON")
        self.window_size = (
            recipe.get("window_size")
            if recipe.get("access") == "sliding_window"
            else None
        )
        self.access = build_access(recipe)
        self.separate_context = separate_context
        if separate_context:
            self.context_key_projection = deepcopy(self.k_proj)
            self.context_value_projection = deepcopy(self.v_proj)

    def _compute_qkv(
        self, draft_hidden_states, context_hidden_states, position_embeddings
    ):
        if not self.separate_context:
            return super()._compute_qkv(
                draft_hidden_states, context_hidden_states, position_embeddings
            )
        batch_size, length = draft_hidden_states.shape[:2]
        query_states = self.q_norm(
            self.q_proj(draft_hidden_states).view(batch_size, length, -1, self.head_dim)
        ).transpose(1, 2)
        key_states = torch.cat(
            (
                self.context_key_projection(context_hidden_states),
                self.k_proj(draft_hidden_states),
            ),
            dim=1,
        )
        value_states = torch.cat(
            (
                self.context_value_projection(context_hidden_states),
                self.v_proj(draft_hidden_states),
            ),
            dim=1,
        )
        key_states = self.k_norm(
            key_states.view(
                batch_size, -1, self.config.num_key_value_heads, self.head_dim
            )
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, -1, self.config.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, *position_embeddings
        )
        return query_states, key_states, value_states
