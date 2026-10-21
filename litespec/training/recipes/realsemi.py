"""RealSemi training registration and immutable experiment metadata."""

from ._dflash_common import DFlashFamilyTrainingAdapter, parse_dflash_family_args


class RealSemiTrainingAdapter(DFlashFamilyTrainingAdapter):
    def training_metadata(self, session):
        return {
            **super().training_metadata(session),
            "realsemi_config": dict(session.draft.config.realsemi_config),
            "loss_config": {
                "ce_loss_alpha": session.args.ce_loss_alpha,
                "l1_loss_alpha": session.args.l1_loss_alpha,
                "loss_decay_gamma": session.args.loss_decay_gamma,
                "objective_chunk_blocks": session.args.objective_chunk_blocks,
                "distribution_loss": "l1_not_half_l1",
            },
        }


ADAPTER = RealSemiTrainingAdapter("realsemi", "RealSemiDraftModel")


def parse_args(*, mode=None, argv=None):
    return parse_dflash_family_args("realsemi", argv)
