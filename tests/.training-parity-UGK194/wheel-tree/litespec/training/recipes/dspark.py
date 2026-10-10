"""DSpark training registration."""

from ._dflash_common import FamilyTrainingAdapter

ADAPTER = FamilyTrainingAdapter("dspark", "DSparkDraftModel")


def parse_args(*, mode=None, argv=None):
    from ._dflash_common import parse_family_args

    return parse_family_args("dspark", argv)
