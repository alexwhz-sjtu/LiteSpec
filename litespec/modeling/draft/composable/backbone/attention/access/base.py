"""Visibility is separate from attention parameterization."""

import torch


def build_visibility(
    inputs, key_positions, key_valid, key_documents, query_length, window_size=None
):
    batch_size, num_anchors = inputs.anchor_positions.shape
    block_size = query_length // num_anchors
    anchor = inputs.anchor_positions.repeat_interleave(block_size, dim=1)
    valid = inputs.block_keep_mask.repeat_interleave(block_size, dim=1)
    documents = inputs.anchor_document_ids
    if documents is None:
        documents = torch.zeros_like(inputs.anchor_positions)
    query_documents = documents.repeat_interleave(block_size, dim=1)
    context = key_positions[:, None] < anchor[:, :, None]
    if window_size is not None:
        context &= key_positions[:, None] >= anchor[:, :, None] - window_size
    context &= key_valid[:, None] & (
        key_documents[:, None] == query_documents[:, :, None]
    )
    block_ids = torch.arange(query_length, device=anchor.device) // block_size
    own_block = block_ids[:, None] == block_ids[None, :]
    current = own_block[None].expand(batch_size, -1, -1)
    current = current & valid[:, None, :]
    return torch.cat((context, current), dim=-1)[:, None] & valid[:, None, :, None]
