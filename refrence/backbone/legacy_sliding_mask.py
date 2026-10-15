# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: legacy_behavior_to_change
# Repository: alexwhz-sjtu/LiteSpec
# Revision: 7fa8b08ddaa29d619dc193d073185e197f77b230
# Source: litespec/core/attention_masks.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/alexwhz-sjtu__LiteSpec/7fa8b08ddaa29d619dc193d073185e197f77b230/litespec/core/attention_masks.py
# License / upstream notices: refrence/licenses/litespec-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Contains legacy block-causal SWA; keep as evidence, do not copy its sliding semantics into the new implementation.


# BEGIN create_dflash_sdpa_mask — upstream lines 58-78
def create_dflash_sdpa_mask(
    anchor_positions, block_keep_mask, S, block_size, device, sliding_window=None
):
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be > 0")
    bsz, num_blocks = anchor_positions.shape
    q_len, kv_len = num_blocks * block_size, S + num_blocks * block_size
    q = torch.arange(q_len, device=device).view(1, 1, -1, 1)
    kv = torch.arange(kv_len, device=device).view(1, 1, 1, -1)
    q_block, q_offset = q // block_size, q % block_size
    anchors = anchor_positions.view(bsz, 1, num_blocks, 1).repeat_interleave(
        block_size, 2
    )
    context_mask = (kv < S) & (kv < anchors)
    if sliding_window is not None:
        context_mask &= kv >= anchors + q_offset - (sliding_window - 1)
    draft_mask = (kv >= S) & (q_block == (kv - S) // block_size)
    if sliding_window is not None:
        draft_mask &= (kv - S) % block_size <= q_offset
    valid = block_keep_mask.view(bsz, 1, num_blocks, 1).repeat_interleave(block_size, 2)
    return (context_mask | draft_mask) & valid

# END create_dflash_sdpa_mask

# BEGIN create_dflash_flex_mask — upstream lines 83-122
def create_dflash_flex_mask(
    anchor_positions,
    block_keep_mask,
    S,
    block_size,
    device,
    flex_backend="TRITON",
    sliding_window=None,
):
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("flex_attention is not available")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be > 0")
    bsz, num_blocks = anchor_positions.shape

    def mask_mod(b, h, q_idx, kv_idx):
        qb, qo = q_idx // block_size, q_idx % block_size
        safe_block_indices = qb.clamp(max=num_blocks - 1)
        anchor = anchor_positions[b, safe_block_indices]
        context_mask = (kv_idx < S) & (kv_idx < anchor)
        if sliding_window is not None:
            context_mask &= kv_idx >= anchor + qo - (sliding_window - 1)
        draft_mask = (kv_idx >= S) & (qb == (kv_idx - S) // block_size)
        if sliding_window is not None:
            draft_mask &= (kv_idx - S) % block_size <= qo
        return (
            (context_mask | draft_mask)
            & block_keep_mask[b, safe_block_indices]
            & (qb < num_blocks)
        )

    return create_flex_block_mask(
        mask_mod,
        B=bsz,
        H=None,
        Q_LEN=num_blocks * block_size,
        KV_LEN=S + num_blocks * block_size,
        device=device,
        backend=flex_backend,
    )

# END create_dflash_flex_mask
