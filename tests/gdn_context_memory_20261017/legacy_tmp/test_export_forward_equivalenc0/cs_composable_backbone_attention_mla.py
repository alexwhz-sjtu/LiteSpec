# Generated inference bundle from composable/backbone/attention/mla.py

"""Expanded MLA; independent latent context projections for DFlare."""
from copy import deepcopy
import torch
from .cs_dflash import Qwen3DFlashMLAAttention, apply_mla_rope
from .cs_composable_backbone_attention_access import build_access
from .cs_composable_backbone_attention_base import AttentionForward

class MLAAttention(AttentionForward, Qwen3DFlashMLAAttention):

    def __init__(self, config, layer_idx, kernels, recipe, separate_context=False):
        super().__init__(config, layer_idx, kernels)
        self.backend = recipe.get('backend', 'sdpa')
        self.flex_backend = recipe.get('flex_backend', 'TRITON')
        self.window_size = recipe.get('window_size') if recipe.get('access') == 'sliding_window' else None
        self.access = build_access(recipe)
        self.separate_context = separate_context
        if separate_context:
            self.context_kv_a_proj = deepcopy(self.kv_a_proj_with_mqa)
            self.context_kv_norm = deepcopy(self.kv_a_layernorm)
            self.context_kv_b_proj = deepcopy(self.kv_b_proj)

    def _compute_qkv(self, draft_hidden_states, context_hidden_states, position_embeddings):
        if not self.separate_context:
            return super()._compute_qkv(draft_hidden_states, context_hidden_states, position_embeddings)
        batch_size, length = draft_hidden_states.shape[:2]
        query_states = self._project_q(draft_hidden_states).view(batch_size, length, self.num_heads, self.qk_head_dim)
        query_nope, query_rope = query_states.split([self.qk_nope_head_dim, self.qk_rope_head_dim], -1)
        latent, context_rope = self.context_kv_a_proj(context_hidden_states).split([self.kv_lora_rank, self.qk_rope_head_dim], -1)
        expanded = self.context_kv_b_proj(self.context_kv_norm(latent)).view(batch_size, -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        context_nope, context_value_states = expanded.split([self.qk_nope_head_dim, self.v_head_dim], -1)
        draft_nope, draft_rope, draft_value = self._project_kv(draft_hidden_states)
        key_nope = torch.cat((context_nope, draft_nope), 1).transpose(1, 2)
        key_rope = torch.cat((context_rope, draft_rope), 1).unsqueeze(1)
        cos, sin = position_embeddings
        query_rope = apply_mla_rope(query_rope.transpose(1, 2), cos[:, -length:], sin[:, -length:], interleaved=self.rope_interleaved)
        key_rope = apply_mla_rope(key_rope, cos, sin, interleaved=self.rope_interleaved).expand(-1, self.num_heads, -1, -1)
        return (torch.cat((query_nope.transpose(1, 2), query_rope), -1), torch.cat((key_nope, key_rope), -1), torch.cat((context_value_states, draft_value), 1).transpose(1, 2))
