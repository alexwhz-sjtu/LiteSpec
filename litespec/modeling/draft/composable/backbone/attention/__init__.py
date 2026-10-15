from .mha import MHAAttention
from .mla import MLAAttention

ATTENTIONS = {"mha": MHAAttention, "mla": MLAAttention}


def build_attention(config, layer_idx, kernels, recipe, separate_context=False):
    return ATTENTIONS[recipe["representation"]](
        config, layer_idx, kernels, recipe, separate_context
    )
