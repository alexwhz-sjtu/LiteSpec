from copy import deepcopy

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from ...dflash import _rope_config
from ...dflash_kernels import DEFAULT_DFLASH_KERNELS
from .attention import build_attention
from .operators import build_local_operator


class DraftLayer(nn.Module):
    def __init__(self, config, layer_idx, recipe):
        super().__init__()
        local = deepcopy(config)
        local.layer_types = ["full_attention"] * config.num_hidden_layers
        local.sliding_window = None
        local._attn_implementation = recipe["attention"].get("backend", "sdpa")
        for key, value in recipe["attention"].items():
            if key in (
                "q_lora_rank",
                "kv_lora_rank",
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "v_head_dim",
                "rope_interleave",
            ):
                setattr(local, key, value)
        local.rope_parameters = deepcopy(
            getattr(local, "rope_parameters", None)
            or local.rope_scaling
            or {"rope_type": "default"}
        )
        kernels = DEFAULT_DFLASH_KERNELS
        self.self_attn = build_attention(
            local,
            layer_idx,
            kernels,
            recipe["attention"],
            config.composable_config["context"]["type"] == "dflare",
        )
        self.rotary_emb = Qwen3RotaryEmbedding(
            _rope_config(local, recipe["attention"]["representation"])
        )
        self.input_norm = kernels.make_rms_norm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_norm = kernels.make_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = kernels.make_mlp(config)
        operator = recipe.get("local_operator", {"type": "none"})
        self.attention_conv = build_local_operator(config, operator)
        self.mlp_conv = build_local_operator(config, operator)
        self.block_size = config.block_size

    def forward(self, draft_hidden_states, context_hidden_states, inputs):
        positions = inputs.context_position_ids
        if positions is None:
            positions = torch.arange(
                context_hidden_states.size(1), device=draft_hidden_states.device
            ).expand(draft_hidden_states.size(0), -1)
        query_positions = (
            inputs.anchor_positions[..., None]
            + torch.arange(self.block_size, device=positions.device)
        ).flatten(1)
        embeddings = self.rotary_emb(
            draft_hidden_states, torch.cat((positions, query_positions), 1)
        )
        residual = draft_hidden_states
        draft_hidden_states = self.input_norm(draft_hidden_states)
        coefficients = None
        if self.attention_conv is not None:
            draft_hidden_states, coefficients = self.attention_conv.prepare(
                draft_hidden_states
            )
        draft_hidden_states = self.self_attn(
            draft_hidden_states, context_hidden_states, embeddings, inputs
        )
        if self.attention_conv is not None:
            draft_hidden_states = self.attention_conv.finish(
                draft_hidden_states, coefficients
            )
        draft_hidden_states = residual + draft_hidden_states
        residual = draft_hidden_states
        draft_hidden_states = self.post_attention_norm(draft_hidden_states)
        if self.mlp_conv is not None:
            draft_hidden_states, coefficients = self.mlp_conv.prepare(
                draft_hidden_states
            )
        draft_hidden_states = self.mlp(draft_hidden_states)
        if self.mlp_conv is not None:
            draft_hidden_states = self.mlp_conv.finish(
                draft_hidden_states, coefficients
            )
        return residual + draft_hidden_states
