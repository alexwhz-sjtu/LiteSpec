import torch

from litespec.core.block_training import (
    apply_exponential_position_decay,
    gather_block_values,
    sample_sorted_anchors,
)


def test_sampler_keeps_anchor_zero_and_appends_physical_padding():
    torch.manual_seed(0)
    anchor_positions, block_keep_mask = sample_sorted_anchors(
        torch.tensor([[True, False, True]]),
        2,
        output_width=4,
        max_valid_anchors=2,
    )
    assert anchor_positions.tolist() == [[0, 2, 0, 0]]
    assert block_keep_mask.tolist() == [[True, True, False, False]]


def test_gather_blocks_reports_bounds():
    values, valid, indices = gather_block_values(
        torch.tensor([[10, 11, 12]]), torch.tensor([[0, 2]]), 2
    )
    assert values.tolist() == [[[10, 11], [12, 12]]]
    assert valid.tolist() == [[[True, True], [True, False]]]
    assert indices.tolist() == [[[0, 1], [2, 2]]]


def test_dlite_decay_uses_semantic_position_one():
    raw = torch.ones(1, 1, 3)
    actual = apply_exponential_position_decay(
        raw, 1.0, torch.tensor([1, 2, 3]), first_prediction_index=1
    )
    torch.testing.assert_close(
        actual, torch.exp(-torch.tensor([0.0, 1.0, 2.0])).view(1, 1, 3)
    )
