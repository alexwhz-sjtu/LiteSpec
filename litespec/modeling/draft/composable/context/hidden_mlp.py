"""DFlash concat -> Linear -> RMSNorm (no nonlinear MLP)."""

from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from ..contracts import ContextOutput
from .base import Context


class HiddenMLPContext(Context):
    def __init__(self, config):
        super().__init__()
        self.depth = config.num_hidden_layers
        self.projection = nn.Linear(
            len(config.composable_config["context"]["target_layer_ids"])
            * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, inputs):
        context_hidden_states = inputs.target_hidden_states.flatten(2)
        context_hidden_states = self.norm(self.projection(context_hidden_states))
        return ContextOutput([context_hidden_states] * self.depth, inputs)
