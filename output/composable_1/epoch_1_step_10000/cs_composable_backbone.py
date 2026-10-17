# Generated inference bundle from composable/backbone/__init__.py

from .cs_composable_backbone_transformer import TransformerBackbone
BACKBONES = {'transformer': TransformerBackbone}

def build_backbone(config):
    return BACKBONES[config.composable_config['backbone'].get('type', 'transformer')](config)
