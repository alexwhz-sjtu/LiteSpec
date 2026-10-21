"""Causal block masks for RealSemi's target-context plus draft KV layout."""

import torch
from .flex_attention import create_flex_block_mask


def serial_mask(
    anchors,
    keep,
    context_length,
    prediction_length,
    mode,
    backend="sdpa",
    flex_backend="TRITON",
):
    batch, blocks = anchors.shape
    length = int(prediction_length)
    q_length = blocks * length

    def allowed(b, h, q, k):
        block = q // length
        safe_block = block.clamp(max=blocks - 1)
        anchor = anchors[b, safe_block]
        if mode == "full":
            context = (k < context_length) & (k < anchor)
        else:
            context = (k < context_length) & (k == block) & (anchor > 0)
        draft_index = k - context_length
        draft = (
            (k >= context_length)
            & (draft_index // length == block)
            & (draft_index % length <= q % length)
        )
        return (context | draft) & keep[b, safe_block] & (block < blocks)

    if backend == "flex_attention":
        return create_flex_block_mask(
            allowed,
            B=batch,
            H=None,
            Q_LEN=q_length,
            KV_LEN=context_length + q_length,
            device=anchors.device,
            backend=flex_backend,
        )
    b = torch.arange(batch, device=anchors.device).view(-1, 1, 1, 1)
    q = torch.arange(q_length, device=anchors.device).view(1, 1, -1, 1)
    k = torch.arange(context_length + q_length, device=anchors.device).view(1, 1, 1, -1)
    return allowed(b, 0, q, k)
