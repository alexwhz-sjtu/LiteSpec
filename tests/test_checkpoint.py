from pathlib import Path

import pytest
import torch

from litespec.modeling.draft.registry import draft_source_files
from litespec.training.checkpoint import (
    load_distributed_training_state,
    ranked_training_state_path,
    save_distributed_training_state,
)


def _state(rank_value: float) -> dict:
    return {
        "training_stage": "test",
        "optimizer_step": 7,
        "optimizer_state_dict": {
            "state": {0: {"exp_avg": torch.full((16,), rank_value)}},
            "param_groups": [{"params": [0]}],
        },
        "scheduler_state_dict": {"last_epoch": 7},
    }


def test_common_training_state_is_metadata_only(tmp_path):
    for rank in range(2):
        save_distributed_training_state(
            str(tmp_path), _state(float(rank)), rank=rank, world_size=2
        )

    common = torch.load(
        tmp_path / "training_state.pt", map_location="cpu", weights_only=False
    )
    assert "optimizer_state_dict" not in common
    assert common["training_stage"] == "test"
    assert common["scheduler_state_dict"] == {"last_epoch": 7}
    assert common["distributed_training_state"]["metadata_only"] is True

    for rank in range(2):
        ranked = Path(ranked_training_state_path(str(tmp_path), rank))
        assert ranked.is_file()
        restored = load_distributed_training_state(
            str(tmp_path), rank=rank, world_size=2
        )
        assert restored["optimizer_state_dict"]["state"][0]["exp_avg"][0] == rank
        assert ranked.stat().st_size > (tmp_path / "training_state.pt").stat().st_size


def test_single_rank_uses_rank_file_and_compact_common_state(tmp_path):
    save_distributed_training_state(
        str(tmp_path), _state(3.0), rank=0, world_size=1
    )

    assert (tmp_path / "training_state_rank_00000.pt").is_file()
    restored = load_distributed_training_state(str(tmp_path), rank=0, world_size=1)
    assert "optimizer_state_dict" in restored


def test_missing_rank_file_does_not_treat_metadata_as_optimizer_state(tmp_path):
    save_distributed_training_state(
        str(tmp_path), _state(0.0), rank=0, world_size=2
    )

    with pytest.raises(FileNotFoundError, match="global rank 1"):
        load_distributed_training_state(str(tmp_path), rank=1, world_size=2)


def test_common_only_checkpoint_is_rejected(tmp_path):
    torch.save(_state(4.0), tmp_path / "training_state.pt")

    with pytest.raises(FileNotFoundError):
        load_distributed_training_state(str(tmp_path), rank=0, world_size=1)


def test_checkpoint_source_files_follow_architecture_dependencies():
    assert set(draft_source_files("DFlashDraftModel")) == {
        "dflash.py", "dflash_kernels.py", "flex_attention.py", "registry.py",
        "sampling.py", "torch_compat.py",
    }
    dflash2 = draft_source_files("DFlash2DraftModel")
    assert "dflash.py" in dflash2
    assert "dflash2.py" in dflash2
    assert "dspark.py" not in dflash2
    assert set(draft_source_files("DSparkDraftModel")) >= {
        "dflash.py",
        "dflash2.py",
        "dspark.py",
        "flex_attention.py",
    }
    assert set(draft_source_files("DLiteDraftModel")) == {
        "dlite.py", "flex_attention.py", "registry.py", "sampling.py", "torch_compat.py",
    }
