# Generated inference bundle from composable/backbone/operators/__init__.py

from .cs_composable_backbone_operators_dflash2_conv import build_dflash2_conv
OPERATORS = {'dflash2_conv': build_dflash2_conv}

def build_local_operator(config, recipe):
    return None if recipe.get('type', 'none') == 'none' else OPERATORS[recipe['type']](config, recipe)
