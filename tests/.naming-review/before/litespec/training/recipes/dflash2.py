"""DFlash2 training registration."""

from ._dflash_common import FamilyTrainingAdapter

ADAPTER = FamilyTrainingAdapter("dflash2", "DFlash2DraftModel")


def parse_args(*, mode=None, argv=None):
    from ._dflash_common import parse_family_args

    return parse_family_args("dflash2", argv)
