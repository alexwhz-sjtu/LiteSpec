# Generated inference bundle from composable/backbone/attention/__init__.py

from .cs_composable_backbone_attention_mha import MHAAttention
from .cs_composable_backbone_attention_mla import MLAAttention
ATTENTIONS = {'mha': MHAAttention, 'mla': MLAAttention}

def build_attention(config, layer_idx, kernels, recipe, separate_context=False):
    return ATTENTIONS[recipe['representation']](config, layer_idx, kernels, recipe, separate_context)
