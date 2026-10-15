from .bv import BVObjective
from .ce import HardCEObjective
from .dpace import DPACEObjective
from .kl import ForwardKLObjective
from .nla import NLAObjective
from .soft_ce import SoftCEObjective
from .tv import TVObjective

OBJECTIVES = {
    "hard_ce": HardCEObjective,
    "soft_ce": SoftCEObjective,
    "forward_kl": ForwardKLObjective,
    "tv": TVObjective,
    "nla": NLAObjective,
    "lk_alpha": NLAObjective,
    "dpace": DPACEObjective,
    "bv": BVObjective,
}


def build_objective(recipe, proposal_type="parallel"):
    return OBJECTIVES[recipe["type"]](recipe, proposal_type)
