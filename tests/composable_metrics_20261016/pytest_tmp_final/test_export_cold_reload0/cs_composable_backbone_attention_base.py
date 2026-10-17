# Generated inference bundle from composable/backbone/attention/base.py

"""Shared noncausal attention, per-layer rotary and confirmed-context cache."""
import torch
from torch.nn import functional as F
from .cs_composable_contracts import LayerState

class AttentionForward:

    def forward(self, draft_hidden_states, context_hidden_states, position_embeddings, inputs):
        query_states, key_states, value_states = self._compute_qkv(draft_hidden_states, context_hidden_states, position_embeddings)
        context_length = context_hidden_states.size(1)
        batch_size = draft_hidden_states.size(0)
        positions = inputs.context_position_ids
        if positions is None:
            positions = torch.arange(context_length, device=draft_hidden_states.device).expand(batch_size, -1)
        valid = inputs.context_valid_mask
        if valid is None:
            valid = torch.ones_like(positions, dtype=torch.bool)
        documents = inputs.document_ids
        if documents is None:
            documents = torch.zeros_like(positions)
        context_key_states, context_value_states = (key_states[:, :, :context_length], value_states[:, :, :context_length])
        if inputs.draft_state is not None:
            previous = inputs.draft_state.layers.get(self.layer_idx)
            if previous is not None:
                if context_length and previous.positions.numel() and (positions[:, :1] <= previous.positions.masked_fill(~previous.valid, -1).amax(-1, keepdim=True)).any():
                    raise ValueError('Cached forward requires strictly new confirmed context positions')
                context_key_states = torch.cat((previous.key_states, context_key_states), dim=2)
                context_value_states = torch.cat((previous.value_states, context_value_states), dim=2)
                positions = torch.cat((previous.positions, positions), dim=1)
                valid = torch.cat((previous.valid, valid), dim=1)
                documents = torch.cat((previous.document_ids, documents), dim=1)
            key_states = torch.cat((context_key_states, key_states[:, :, context_length:]), dim=2)
            value_states = torch.cat((context_value_states, value_states[:, :, context_length:]), dim=2)
            cache_key, cache_value, cache_positions, cache_valid, cache_documents = (context_key_states, context_value_states, positions, valid, documents)
            cache_valid = valid & (positions < inputs.anchor_positions[:, :1])
            if self.window_size is not None:
                width = min(self.window_size, positions.size(1))
                priority = positions.masked_fill(~cache_valid, -1)
                indices = priority.argsort(dim=1, descending=True)[:, :width]
                indices = indices.flip(-1)
                cache_key = context_key_states.gather(2, indices[:, None, :, None].expand(-1, context_key_states.size(1), -1, context_key_states.size(-1)))
                cache_value = context_value_states.gather(2, indices[:, None, :, None].expand(-1, context_value_states.size(1), -1, context_value_states.size(-1)))
                cache_positions = positions.gather(1, indices)
                cache_valid = cache_valid.gather(1, indices)
                cache_documents = documents.gather(1, indices)
            inputs.draft_state.layers[self.layer_idx] = LayerState(cache_key.detach(), cache_value.detach(), cache_positions, cache_valid, cache_documents)
        visibility = self.access(inputs, positions, valid, documents, draft_hidden_states.size(1), **{'window_size': self.window_size} if self.window_size is not None else {})
        repeats = query_states.size(1) // key_states.size(1)
        if repeats != 1:
            key_states = key_states.repeat_interleave(repeats, dim=1)
            value_states = value_states.repeat_interleave(repeats, dim=1)
        if self.backend == 'eager':
            scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
            scores = scores.masked_fill(~visibility, float('-inf'))
            attention_probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype).nan_to_num()
            attention_probabilities = F.dropout(attention_probabilities, p=self.attention_dropout, training=self.training)
            output = torch.matmul(attention_probabilities.to(value_states.dtype), value_states)
        elif self.backend == 'flex_attention':
            from .cs_flex_attention import FLASH_KV_TILE, FLASH_Q_TILE, TRITON_MASK_TILE, compile_friendly_flex_attention, create_flex_block_mask, flex_kernel_options
            q_tile, kv_tile = (FLASH_Q_TILE, FLASH_KV_TILE) if self.flex_backend.upper() == 'FLASH' else (TRITON_MASK_TILE, TRITON_MASK_TILE)
            flat_visibility = F.pad(visibility[:, 0], (0, -visibility.size(3) % kv_tile, 0, -visibility.size(2) % q_tile), value=False)

            def mask_mod(b, h, q, k):
                return flat_visibility[b, q, k]
            mask = create_flex_block_mask(mask_mod, B=query_states.size(0), H=None, Q_LEN=query_states.size(2), KV_LEN=key_states.size(2), device=str(query_states.device), backend=self.flex_backend)
            output = compile_friendly_flex_attention(query_states, key_states, value_states, block_mask=mask, scale=self.scaling, kernel_options=flex_kernel_options(self.flex_backend))
        else:
            output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=visibility, dropout_p=self.attention_dropout if self.training else 0.0, is_causal=False, scale=self.scaling)
        output = output.transpose(1, 2).reshape(*draft_hidden_states.shape[:2], -1)
        return self.o_proj(output) * visibility.any(-1).transpose(1, 2).to(output.dtype)
