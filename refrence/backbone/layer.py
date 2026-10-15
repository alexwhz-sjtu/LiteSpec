# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: author_repository
# Repository: z-lab/dflash
# Revision: 07ebd93db9f472af339b644bb70221ad8428328a
# Source: dflash/model.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/z-lab__dflash/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py
# License / upstream notices: refrence/licenses/dflash-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Preserve residual/norm/conv order; kernel scheduling is separate from mathematical structure.


# BEGIN Qwen3DFlashDecoderLayer — upstream lines 423-475
class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_conv = None
        self.mlp_conv = None

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_kernel = None
        if self.attention_conv is not None:
            hidden_states, attention_kernel = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        if attention_kernel is not None:
            hidden_states = self.attention_conv.finish(hidden_states, attention_kernel)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_kernel = None
        if self.mlp_conv is not None:
            hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if mlp_kernel is not None:
            hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)
        hidden_states = residual + hidden_states
        return hidden_states

# END Qwen3DFlashDecoderLayer
