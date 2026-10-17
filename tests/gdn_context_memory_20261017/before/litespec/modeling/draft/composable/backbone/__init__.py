from .transformer import TransformerBackbone

BACKBONES = {"transformer": TransformerBackbone}


def build_backbone(config):
    return BACKBONES[config.composable_config["backbone"].get("type", "transformer")](
        config
    )
