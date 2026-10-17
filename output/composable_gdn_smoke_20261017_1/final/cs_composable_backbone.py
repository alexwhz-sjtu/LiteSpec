# Generated inference bundle from composable/backbone/__init__.py

from .cs_composable_backbone_transformer import TransformerBackbone
from .cs_composable_backbone_gdn import GDNBackbone
BACKBONES = {'transformer': TransformerBackbone, 'gdn_context_memory': GDNBackbone}

def build_backbone(config):
    return BACKBONES[config.composable_config['backbone'].get('type', 'transformer')](config)
