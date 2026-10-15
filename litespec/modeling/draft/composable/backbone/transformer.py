from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from ..contracts import BackboneOutput
from .base import Backbone
from .layer import DraftLayer


class TransformerBackbone(Backbone):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DraftLayer(config, index, recipe)
                for index, recipe in enumerate(
                    config.composable_config["backbone"]["layers"]
                )
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

    def forward(self, context):
        draft_hidden_states = context.inputs.draft_embeddings
        for layer, context_hidden_states in zip(
            self.layers, context.context_hidden_states
        ):
            if self.gradient_checkpointing and self.training:
                draft_hidden_states = checkpoint(
                    layer,
                    draft_hidden_states,
                    context_hidden_states,
                    context.inputs,
                    use_reentrant=False,
                )
            else:
                draft_hidden_states = layer(
                    draft_hidden_states, context_hidden_states, context.inputs
                )
        return BackboneOutput(
            self.norm(draft_hidden_states), context.inputs.draft_state
        )
