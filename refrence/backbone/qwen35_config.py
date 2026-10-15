# REFERENCE EXCERPT ONLY; never import into training or deployment.
# Classification: official_model_implementation
# Repository: huggingface/transformers
# Revision: aad13b87ed59f2afcfaebc985f403301887a35fc (v5.3.0 reference baseline, not a dependency upgrade)
# Source: src/transformers/models/qwen3_5/configuration_qwen3_5.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/huggingface__transformers/aad13b87ed59f2afcfaebc985f403301887a35fc/src/transformers/models/qwen3_5/configuration_qwen3_5.py
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

# BEGIN Qwen3_5TextConfig — upstream lines 24-197
class Qwen3_5TextConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3_5TextModel`]. It is used to instantiate a
    Qwen3_5 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of
    Qwen3.5-9B-Instruct [Qwen/Qwen3.5-9B-Instruct](https://huggingface.co/Qwen/Qwen3.5-9B-Instruct).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 248320):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            `inputs_ids`.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 12288):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        head_dim (`int`, *optional*, defaults to 256):
            Projection weights dimension in multi-head attention.
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Kernel size of the convolution used in linear attention layers.
        linear_key_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each key head in linear attention.
        linear_value_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each value head in linear attention.
        linear_num_key_heads (`int`, *optional*, defaults to 16):
            Number of key heads used in linear attention layers.
        linear_num_value_heads (`int`, *optional*, defaults to 32):
            Number of value heads used in linear attention layers.
        layer_types (`list[str]`, *optional*):
            Types of each layer (attention or linear).
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*):
            End of stream token id.

    ```python
    >>> from transformers import Qwen3_5TextModel, Qwen3_5TextConfig

    >>> # Initializing a Qwen3.5 style configuration
    >>> configuration =  Qwen3_5TextConfig()

    >>> # Initializing a model from the Qwen3.5-9B style configuration
    >>> model = Qwen3_5TextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "qwen3_5_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        layer_types=None,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ):
        kwargs["ignore_keys_at_rope_validation"] = {"mrope_section", "mrope_interleaved"}
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.rope_parameters = rope_parameters
        kwargs.setdefault("partial_rotary_factor", 0.25)  # assign default for BC

        self.layer_types = layer_types
        if self.layer_types is None:
            interval_pattern = kwargs.get("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        # linear attention part
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        super().__init__(**kwargs)

# END Qwen3_5TextConfig
