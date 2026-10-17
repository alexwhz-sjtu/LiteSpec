# Generated inference bundle from composable/context/__init__.py

from .cs_composable_context_hidden_gated import HiddenGatedContext
from .cs_composable_context_hidden_mlp import HiddenMLPContext
CONTEXTS = {'dflash': HiddenMLPContext, 'dflare': HiddenGatedContext}

def build_context(config):
    return CONTEXTS[config.composable_config['context']['type']](config)
