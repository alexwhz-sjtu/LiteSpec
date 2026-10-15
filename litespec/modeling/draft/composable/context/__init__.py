from .hidden_gated import HiddenGatedContext
from .hidden_mlp import HiddenMLPContext

CONTEXTS = {"dflash": HiddenMLPContext, "dflare": HiddenGatedContext}


def build_context(config):
    return CONTEXTS[config.composable_config["context"]["type"]](config)
