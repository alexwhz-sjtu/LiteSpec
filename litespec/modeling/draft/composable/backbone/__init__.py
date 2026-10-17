from .transformer import TransformerBackbone
from .gdn import GDNBackbone

BACKBONES = {"transformer": TransformerBackbone, "gdn_context_memory": GDNBackbone}


def build_backbone(config):
    return BACKBONES[config.composable_config["backbone"].get("type", "transformer")](
        config
    )
