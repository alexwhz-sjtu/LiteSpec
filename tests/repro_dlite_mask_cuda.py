"""Isolate DLite's mask construction from model/optimizer execution.

Run from the repository root, for example::

    PYTHONPATH=. .venv/bin/python tests/repro_dlite_mask_cuda.py

Each CUDA stage is synchronized so an asynchronous failure is attributed to
the stage that launched it. No training data or checkpoint is required.
"""

from __future__ import annotations

import argparse

import torch
from torch.nn.attention.flex_attention import create_block_mask

from litespec.core.attention_masks import create_dlite_flex_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--widths", type=int, nargs="+", default=[768, 767, 16, 1, 17, 129]
    )
    parser.add_argument("--query-length", type=int, default=8)
    parser.add_argument("--chs-slots", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if min(*args.widths, args.query_length, args.chs_slots, args.batch_size) < 1:
        parser.error("all dimensions must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    device = torch.device("cuda", torch.cuda.current_device())
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"gpu={torch.cuda.get_device_name(device)}",
        flush=True,
    )
    for width in args.widths:
        anchors = torch.arange(width, device=device).view(1, -1).repeat(args.batch_size, 1)
        keep = torch.ones_like(anchors, dtype=torch.bool)
        # Include padded anchors and match prepare_batch's expanded token mask.
        if width > 1:
            keep[:, -1] = False
        token_keep = keep.unsqueeze(-1).expand(-1, -1, 1)
        torch.cuda.synchronize(device)
        print(f"width={width}: constructing mask", flush=True)
        mask = create_dlite_flex_mask(
            model_role="student",
            anchor_positions=anchors,
            block_keep_mask=keep,
            token_keep_mask=token_keep,
            seq_len=4096,
            swa_window_size=1,
            chs_slots=args.chs_slots,
            query_len=args.query_length,
            device=device,
            flex_backend="TRITON",
        )
        torch.cuda.synchronize(device)
        print(f"width={width}: mask completed", flush=True)
        # This is the kind of host-to-device operation where the remote trace
        # reported the error; it should be harmless after a successful mask.
        torch.tensor(list(range(args.chs_slots)), device=device, dtype=torch.long)
        torch.cuda.synchronize(device)
        # Bound eager intermediate memory; larger widths still check CUDA safety.
        if width <= 129:
            reference = create_block_mask(
                mask.mask_mod,
                B=args.batch_size,
                H=None,
                Q_LEN=width * args.query_length,
                KV_LEN=width * (args.chs_slots + args.query_length),
                device=device,
            )
            torch.cuda.synchronize(device)
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
                    msg=lambda message, name=name: f"{name}: {message}",
                )
        print(f"width={width}: PASS", flush=True)


if __name__ == "__main__":
    main()
