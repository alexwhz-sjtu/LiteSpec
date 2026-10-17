"""Thin deployable orchestration of context, backbone and proposal."""

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel

from ..dflash import build_target_layer_ids
from ..dflash2 import DFlashGroupedConv
from ..registry import register_draft
from .backbone import build_backbone
from .backbone.attention.gated_deltanet import GDNContextMemoryMixer
from .configuration import ComposableDraftConfig
from .context import build_context
from .context.hidden_gated import HiddenGatedContext
from .proposal import build_proposal


@register_draft
class ComposableDraftModel(Qwen3PreTrainedModel):
    config_class = ComposableDraftConfig
    method_config_key = "composable_config"
    _no_split_modules = ["DraftLayer", "GDNLayer"]
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        config.validate_recipe()
        recipe = config.composable_config
        if recipe["context"].get("target_layer_ids") is None:
            recipe["context"]["target_layer_ids"] = (
                list(range(config.num_target_layers))
                if config.num_target_layers < 4
                else sorted(
                    set(
                        build_target_layer_ids(
                            config.num_target_layers, config.num_hidden_layers
                        )
                    )
                )
            )
        config.validate_recipe()
        self.target_layer_ids = recipe["context"]["target_layer_ids"]
        self.block_size = config.block_size
        self.mask_token_id = recipe.get("mask_token_id")
        self.context = build_context(config)
        self.backbone = build_backbone(config)
        self.proposal = build_proposal(config)
        self.post_init()
        # post_init initializes Linear modules; restore source-specific identity starts.
        for module in self.modules():
            if isinstance(module, DFlashGroupedConv):
                torch.nn.init.zeros_(module.kernel_projection.weight)
            if isinstance(module, HiddenGatedContext):
                module.reset_fusion_prior()
            if isinstance(module, GDNContextMemoryMixer):
                module.reset_gdn_parameters()
        head = getattr(self.proposal, "markov_head", None)
        if head is not None:
            torch.nn.init.zeros_(head.markov_w2.weight)
        selector = getattr(self.proposal, "candidate_selector", None)
        if selector is not None:
            torch.nn.init.zeros_(selector.successor_codebook)

    def forward(self, inputs, *, shared_ops=None):
        if inputs.mode not in ("features", "teacher_forcing", "sample"):
            raise ValueError("mode must be features, teacher_forcing, or sample")
        if inputs.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.training and inputs.draft_state is not None:
            raise ValueError("Persistent inference state is not accepted in training")
        if inputs.draft_state is not None and inputs.anchor_positions.size(1) != 1:
            raise ValueError("Cached inference expects one current block per sample")
        context = self.context(inputs)
        features = self.backbone(context)
        output = self.proposal(features, context, shared_ops)
        output.backbone_hidden_states = features.draft_hidden_states
        return output
