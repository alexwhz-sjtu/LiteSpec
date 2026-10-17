# Generated inference bundle from composable/backbone/operators/dflash2_conv.py

"""Use the existing source-matched dynamic convolution without formula changes."""
from .cs_dflash2 import DFlashGroupedConv

def build_dflash2_conv(config, recipe):
    return DFlashGroupedConv(config.hidden_size, config.block_size, int(recipe.get('kernel_size', 2)), int(recipe.get('group_size', 16)))
