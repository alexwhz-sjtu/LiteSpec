"""Parallel DFlash features followed by token-conditioned causal refinement."""

from __future__ import annotations

import time
import torch
from torch import nn
from transformers import DynamicCache
from .dflash import (
    DFlashDraftModel,
    extract_context_feature,
    target_embed_tokens,
    target_lm_head,
    rotate_half,
)
from .sampling import sample
from .registry import method_config, register_draft
from .realsemi_masks import serial_mask


class RealSemiSerialCache:
    """Immutable projected target KV plus block-local, growing draft KV.

    Serial attention projects only the current draft input. Context is never
    passed through Cache.update, so it cannot be appended more than once.
    """

    def __init__(self, context):
        self.context = dict(context)
        self.draft = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx in self.draft:
            old_k, old_v = self.draft[layer_idx]
            key_states = torch.cat((old_k, key_states), dim=-2)
            value_states = torch.cat((old_v, value_states), dim=-2)
        self.draft[layer_idx] = (key_states, value_states)
        context_k, context_v = self.context[layer_idx]
        return (
            torch.cat((context_k, key_states), dim=-2),
            torch.cat((context_v, value_states), dim=-2),
        )


@register_draft
class RealSemiDraftModel(DFlashDraftModel):
    method_config_key = "realsemi_config"

    def __init__(self, config, dflash_kernels=None):
        settings = method_config(config)
        self.num_parallel_layers = int(settings.get("num_parallel_layers", 4))
        self.serial_context_mode = str(settings.get("serial_context_mode", "full"))
        if not 1 <= self.num_parallel_layers < int(config.num_hidden_layers):
            raise ValueError(
                "RealSemi requires 1 <= num_parallel_layers < num_hidden_layers"
            )
        if self.serial_context_mode not in ("full", "last"):
            raise ValueError("serial_context_mode must be full or last")
        if settings.get("attention_mode", "gqa") not in ("gqa", "mha"):
            raise ValueError("RealSemi supports GQA/MHA")
        super().__init__(config, dflash_kernels)
        self.token_hidden_fusion = nn.Linear(
            2 * config.hidden_size, config.hidden_size, bias=False
        )
        # Initialize only the new projection; do not reinitialize inherited layers.
        self._init_weights(self.token_hidden_fusion)

    def _parallel(self, hidden, context, position_ids, mask=None, cache=None, **kwargs):
        rotary = self.rotary_emb(hidden, position_ids)
        for layer in self.layers[: self.num_parallel_layers]:
            hidden = layer(
                hidden_states=hidden,
                target_hidden=context,
                attention_mask=mask,
                position_ids=position_ids,
                position_embeddings=rotary,
                past_key_value=cache,
                use_cache=cache is not None,
                **kwargs,
            )
        return hidden

    def forward_parallel(
        self,
        *,
        noise_embedding,
        target_hidden,
        position_ids,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        context = self.hidden_norm(self.fc(target_hidden))
        return self._parallel(
            noise_embedding,
            context,
            position_ids,
            attention_mask,
            past_key_values,
            **kwargs,
        )

    def forward(
        self,
        *,
        noise_embedding,
        target_hidden,
        position_ids,
        previous_token_embeddings,
        anchor_positions,
        block_keep_mask,
        attention_mask=None,
        flex_backend="TRITON",
        **kwargs,
    ):
        """Teacher-forced full-block execution; future tokens never enter z."""
        batch, blocks = anchor_positions.shape
        length = self.block_size - 1
        context = self.hidden_norm(self.fc(target_hidden))
        z = self._parallel(
            noise_embedding, context, position_ids, attention_mask, **kwargs
        )
        z = z.reshape(batch, blocks, self.block_size, -1)[:, :, 1:]
        hidden = self.token_hidden_fusion(torch.cat((previous_token_embeddings, z), -1))
        hidden = hidden.reshape(batch, blocks * length, -1)
        query_positions = (
            anchor_positions[..., None]
            + torch.arange(1, self.block_size, device=z.device)
        ).reshape(batch, -1)
        if self.serial_context_mode == "last":
            context_positions = (anchor_positions - 1).clamp_min(0)
            context = context.gather(
                1, context_positions[..., None].expand(-1, -1, context.size(-1))
            )
        else:
            context_positions = torch.arange(context.size(1), device=z.device)[
                None
            ].expand(batch, -1)
        serial_positions = torch.cat((context_positions, query_positions), -1)
        mask = serial_mask(
            anchor_positions,
            block_keep_mask,
            context.size(1),
            length,
            self.serial_context_mode,
            self.config._attn_implementation,
            flex_backend,
        )
        rotary = self.rotary_emb(hidden, serial_positions)
        for layer in self.layers[self.num_parallel_layers :]:
            hidden = layer(
                hidden_states=hidden,
                target_hidden=context,
                attention_mask=mask,
                position_ids=serial_positions,
                position_embeddings=rotary,
                **kwargs,
            )
        return self.norm(hidden).reshape(batch, blocks, length, -1)

    def prepare_serial_cache(
        self, target_hidden, target_position_ids, context_cache=None
    ):
        """Append newly verified target features, or replace with the last one.

        context_cache persists across verification rounds; returned draft cache
        is always empty. Callers pass only the newly verified target features.
        """
        context_cache = {} if context_cache is None else context_cache
        if self.serial_context_mode == "last":
            target_hidden = target_hidden[:, -1:]
            target_position_ids = target_position_ids[:, -1:]
        context = self.hidden_norm(self.fc(target_hidden))
        cos, sin = self.rotary_emb(context, target_position_ids)
        for index in range(self.num_parallel_layers, len(self.layers)):
            attention = self.layers[index].self_attn
            key = attention.k_proj(context).view(
                context.size(0), context.size(1), -1, attention.head_dim
            )
            key = attention.k_norm(key).transpose(1, 2)
            value = (
                attention.v_proj(context)
                .view(context.size(0), context.size(1), -1, attention.head_dim)
                .transpose(1, 2)
            )
            key = key * cos[:, None] + rotate_half(key) * sin[:, None]
            if self.serial_context_mode == "full" and index in context_cache:
                old_k, old_v = context_cache[index]
                key, value = torch.cat((old_k, key), -2), torch.cat((old_v, value), -2)
            context_cache[index] = (key, value)
        return RealSemiSerialCache(context_cache)

    def serial_step(
        self, previous_token_embeddings, parallel_hidden, position_ids, cache
    ):
        hidden = self.token_hidden_fusion(
            torch.cat((previous_token_embeddings, parallel_hidden), -1)
        )
        rotary = self.rotary_emb(hidden, position_ids)
        # Only the current draft input is projected. All returned cached keys are
        # causally visible, so a single-token query needs no triangular mask.
        empty_context = hidden[:, :0]
        for layer in self.layers[self.num_parallel_layers :]:
            hidden = layer(
                hidden_states=hidden,
                target_hidden=empty_context,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=rotary,
                past_key_value=cache,
                use_cache=True,
                is_causal=False,
            )
        return self.norm(hidden)

    @torch.inference_mode()
    def spec_generate(
        self,
        target,
        input_ids,
        max_new_tokens,
        stop_token_ids,
        temperature,
        top_p=1.0,
        top_k=0,
        target_cache=None,
    ):
        """Greedy draft with target-distribution-preserving match verification."""
        if input_ids.size(0) != 1:
            raise ValueError("RealSemi speculative generation requires batch size one")
        if temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
            raise ValueError("Invalid sampling parameters")
        self.eval()
        initial = input_ids.size(1)
        limit = initial + max_new_tokens
        if max_new_tokens <= 0:
            return input_ids
        output_ids = torch.full(
            (1, limit + self.block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        output_ids[:, :initial] = input_ids
        positions = torch.arange(output_ids.size(1), device=input_ids.device)[None]
        target_kv = DynamicCache() if target_cache is None else target_cache
        parallel_kv = DynamicCache()
        serial_context = {}
        output = target(
            input_ids,
            position_ids=positions[:, :initial],
            past_key_values=target_kv,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        output_ids[:, initial : initial + 1] = sample(
            output.logits, temperature, top_p, top_k
        )
        target_hidden = extract_context_feature(
            output.hidden_states, self.target_layer_ids
        )
        start = initial
        accept_lengths = []
        if input_ids.is_cuda:
            torch.cuda.synchronize(input_ids.device)
        begin = time.perf_counter()
        while start < limit:
            block_ids = output_ids[:, start : start + self.block_size].clone()
            previous_context_length = parallel_kv.get_seq_length()
            context_positions = positions[:, previous_context_length:start]
            z = self.forward_parallel(
                noise_embedding=target_embed_tokens(target)(block_ids),
                target_hidden=target_hidden,
                position_ids=positions[
                    :, previous_context_length : start + self.block_size
                ],
                past_key_values=parallel_kv,
                is_causal=False,
            )
            parallel_kv.crop(start)
            serial_kv = self.prepare_serial_cache(
                target_hidden, context_positions, serial_context
            )
            previous = block_ids[:, :1]
            for offset in range(1, self.block_size):
                hidden = self.serial_step(
                    target_embed_tokens(target)(previous),
                    z[:, offset : offset + 1],
                    positions[:, start + offset : start + offset + 1],
                    serial_kv,
                )
                previous = sample(target_lm_head(target)(hidden), 0.0)
                block_ids[:, offset : offset + 1] = previous
            verify_width = min(self.block_size, limit - start)
            verified_ids = block_ids[:, :verify_width]
            output = target(
                verified_ids,
                position_ids=positions[:, start : start + verify_width],
                past_key_values=target_kv,
                use_cache=True,
                output_hidden_states=True,
            )
            posterior = sample(output.logits, temperature, top_p, top_k)
            accepted = int(
                (verified_ids[:, 1:] == posterior[:, :-1]).cumprod(-1).sum().item()
            )
            advance = accepted + 1
            output_ids[:, start : start + advance] = verified_ids[:, :advance]
            output_ids[:, start + advance] = posterior[:, accepted]
            start += advance
            target_kv.crop(start)
            target_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )[:, :advance]
            accept_lengths.append(advance)
            if (
                stop_token_ids
                and torch.isin(
                    output_ids[:, initial : start + 1],
                    torch.tensor(stop_token_ids, device=input_ids.device),
                ).any()
            ):
                break
        result = output_ids[:, : min(start + 1, limit)]
        if stop_token_ids:
            stops = torch.isin(
                result[0, initial:],
                torch.tensor(stop_token_ids, device=input_ids.device),
            ).nonzero()
            if stops.numel():
                result = result[:, : initial + int(stops[0, 0]) + 1]
        if input_ids.is_cuda:
            torch.cuda.synchronize(input_ids.device)
        self._last_decode_stats = {
            "accept_lengths": accept_lengths,
            "decode_wall_time": time.perf_counter() - begin,
        }
        return result
