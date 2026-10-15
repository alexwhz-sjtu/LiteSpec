from .dflash2_conv import build_dflash2_conv

OPERATORS = {"dflash2_conv": build_dflash2_conv}


def build_local_operator(config, recipe):
    return (
        None
        if recipe.get("type", "none") == "none"
        else OPERATORS[recipe["type"]](config, recipe)
    )
