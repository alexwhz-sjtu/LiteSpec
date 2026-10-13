from unittest import mock

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

from litespec.core.attention_masks import (
    create_dlite_flex_mask,
    flash_kv_padding,
    flash_padded_block_width,
)
from litespec.modeling.draft.flex_attention import (
    create_flex_block_mask,
    flex_kernel_options,
)


def test_flash_v1_shape_alignment():
    width = flash_padded_block_width(78, query_len=8, chs_slots=14)
    assert width == 128
    assert (width * 8) % 256 == 0
    assert (width * 22) % 128 == 0
    assert flash_padded_block_width(512, 8, 14) == 512


def test_teacher_history_padding_only_extends_kv_tail():
    assert flash_kv_padding(4096 + 78 * 22) == 76
    assert flash_kv_padding(4096 + 128 * 22) == 0


def test_triton_kernel_option_is_version_safe():
    options = flex_kernel_options("TRITON")
    assert "BACKEND" in options or "FORCE_USE_FLEX_ATTENTION" in options


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        flex_kernel_options("auto")


def test_block_mask_creation_uses_one_backend_policy():
    def mask_mod(b, h, q, kv):
        del b, h
        return q >= kv

    with (
        mock.patch(
            "litespec.modeling.draft.flex_attention.flex_kernel_options",
            return_value={"BACKEND": "FLASH"},
        ),
        mock.patch(
            "litespec.modeling.draft.flex_attention.create_block_mask"
        ) as direct,
        mock.patch(
            "litespec.modeling.draft.flex_attention.compile_friendly_create_block_mask"
        ) as compiled,
    ):
        create_flex_block_mask(
            mask_mod,
            B=1,
            H=None,
            Q_LEN=16,
            KV_LEN=32,
            device="cpu",
            backend="FLASH",
        )
        assert direct.call_args.kwargs["BLOCK_SIZE"] == (256, 128)
        compiled.assert_not_called()

        create_flex_block_mask(
            mask_mod,
            B=1,
            H=None,
            Q_LEN=256,
            KV_LEN=512,
            device="cpu",
            backend="TRITON",
        )
        compiled.assert_called_once()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dlite_mask_shape_transitions_match_eager():
    # The old compiled path produced corrupt q_num_blocks at width=16 and
    # kv_num_blocks at width=1 after compiling larger, dynamic shapes.
    for width in (768, 767, 16, 1, 17, 129):
        anchors = torch.arange(width, device="cuda").view(1, -1)
        keep = torch.ones_like(anchors, dtype=torch.bool)
        if width > 1:
            keep[:, -1] = False
        mask = create_dlite_flex_mask(
            model_role="student",
            anchor_positions=anchors,
            block_keep_mask=keep,
            token_keep_mask=keep.unsqueeze(-1),
            seq_len=10240,
            swa_window_size=1,
            chs_slots=12,
            query_len=8,
            device="cuda",
        )
        torch.cuda.synchronize()
        if width > 129:
            continue
        reference = create_block_mask(
            mask.mask_mod,
            B=1,
            H=None,
            Q_LEN=width * 8,
            KV_LEN=width * 20,
            device="cuda",
        )
        for name in (
            "kv_num_blocks",
            "kv_indices",
            "full_kv_num_blocks",
            "full_kv_indices",
            "q_num_blocks",
            "q_indices",
            "full_q_num_blocks",
            "full_q_indices",
        ):
            torch.testing.assert_close(
                getattr(mask, name),
                getattr(reference, name),
                msg=lambda message, name=name: f"width={width}, {name}: {message}",
            )
