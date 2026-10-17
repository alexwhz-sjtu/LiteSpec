# Generated inference bundle from composable/context/hidden_gated.py

"""DFlare static per-layer softmax fusion; Tencent/AngelSpec d3412bed."""
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from .cs_composable_contracts import ContextOutput
from .cs_composable_context_base import Context

class HiddenGatedContext(Context):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.target_count = len(config.composable_config['context']['target_layer_ids'])
        self.layer_fusion_weights = nn.Parameter(torch.empty(config.num_hidden_layers, self.target_count))
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.reset_fusion_prior()

    def reset_fusion_prior(self):
        depth, count = self.layer_fusion_weights.shape
        with torch.no_grad():
            for index in range(depth):
                preferred = round(index / max(depth - 1, 1) * (count - 1))
                self.layer_fusion_weights[index].copy_(-2.0 * (torch.arange(count, device=self.layer_fusion_weights.device) - preferred).abs())

    def forward(self, inputs):
        target_hidden_states = inputs.target_hidden_states.reshape(*inputs.target_hidden_states.shape[:2], self.target_count, self.hidden_size)
        weights = self.layer_fusion_weights.softmax(-1).to(target_hidden_states.dtype)
        return ContextOutput([self.norm(torch.einsum('t,bstd->bsd', weight, target_hidden_states)) for weight in weights], inputs)
