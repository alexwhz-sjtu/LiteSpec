# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: author_repository
# Repository: z-lab/dflash
# Revision: 07ebd93db9f472af339b644bb70221ad8428328a
# Source: dflash/model.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/z-lab__dflash/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py
# License / upstream notices: refrence/licenses/dflash-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Capture concat+fc+hidden_norm and model forward; only the context operations will be migrated.


# BEGIN extract_context_feature — upstream lines 37-43
def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: list[int] | None,
) -> torch.Tensor:
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected_states, dim=-1)

# END extract_context_feature

# BEGIN DFlashDraftModel — upstream lines 550-627
class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules: ClassVar[list[str]] = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = _draft_value(
            config,
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = int(_draft_value(config, "block_size", 16))
        self.mask_token_id = _draft_value(config, "mask_token_id")
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        noise_embedding: torch.Tensor | None = None,
        target_hidden: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    def compute_logits(self, hidden, output_head):
        logits = output_head(hidden)
        logits = logits * float(_draft_value(self.config, "output_multiplier", 1.0))
        softcap = _draft_value(self.config, "final_logit_softcapping")
        if softcap is not None and float(softcap) > 0:
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
        return logits

    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ):
        self.eval()
        return dflash_generate(
            self,
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

# END DFlashDraftModel
