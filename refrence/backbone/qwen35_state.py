# REFERENCE EXCERPT ONLY; never import into training or deployment.
# Classification: official_model_implementation
# Repository: huggingface/transformers
# Revision: aad13b87ed59f2afcfaebc985f403301887a35fc (v5.3.0 reference baseline, not a dependency upgrade)
# Source: src/transformers/models/qwen3_5/modeling_qwen3_5.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/huggingface__transformers/aad13b87ed59f2afcfaebc985f403301887a35fc/src/transformers/models/qwen3_5/modeling_qwen3_5.py
# License: refrence/licenses/qwen35-transformers-LICENSE
# Original symbol names and bodies retained; imports/dependencies live in the full source.

# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# BEGIN Qwen3_5DynamicCache — upstream lines 69-156
class Qwen3_5DynamicCache:
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the linear attention
    cache (which has a constant shape regardless of seq_len).

    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
    and `ssm_states` for gated deltanet cache. Each of these lists has `num_layers` tensors. The expected shape for each tensor
    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
    while `conv_states` and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).
    For linear attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` represents the convolution state and has a shape of `(batch_size, d_inner, d_conv)`,
    and `recurrent_states` represents the recurrent state and has a shape of `(batch_size, d_inner, d_state)`.
    """

    is_compileable = False

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.layer_types = config.layer_types
        self.transformer_layers = [
            i for i in range(config.num_hidden_layers) if self.layer_types[i] == "full_attention"
        ]
        self.last_linear_layer = len(self.layer_types) - 1 - self.layer_types[::-1].index("linear_attention")

        # Initialize everything to None -> will be lazy initialized to allow multi-gpu (device_map) inference
        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]

    def __len__(self):
        return len(self.layer_types)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                device = self.key_cache[layer_idx].device
                beam_idx = beam_idx.to(device)
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx)

            if self.conv_states[layer_idx] is not None:
                device = self.conv_states[layer_idx].device
                beam_idx = beam_idx.to(device)
                self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx)
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, beam_idx)

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
        """
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    @property
    def has_previous_state(self):
        """We have a previous state if the last linear (conv) layer was already updated."""
        return self.conv_states[self.last_linear_layer] is not None

# END Qwen3_5DynamicCache
