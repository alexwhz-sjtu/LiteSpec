"""Integer event topology; all trainable tensors stay inside model forward.

Write events follow real context order. Read events precede the write at the
same absolute position and have an identity state transition. This permits a
single differentiable chunk scan without replicating any anchor's prefix.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GDNEventLayout:
    context_indices: torch.Tensor
    query_indices: torch.Tensor
    write_events: torch.Tensor
    read_events: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_seqlens_cpu: torch.Tensor
    context_lengths: tuple[int, ...]
    samples: tuple[int, ...]
    num_events: int


def build_event_layout(inputs, block_size):
    """Build once per forward; metadata work is per document, not per anchor."""
    device = inputs.draft_embeddings.device
    n, a = inputs.anchor_positions.shape
    if inputs.draft_embeddings.shape[:2] != (n, a * block_size):
        raise ValueError("draft_embeddings must have shape [N,A*block_size,D]")
    s = inputs.target_hidden_states.size(1)
    positions = (
        torch.arange(s).expand(n, -1)
        if inputs.context_position_ids is None
        else inputs.context_position_ids.detach().cpu()
    )
    valid = (
        torch.ones(n, s, dtype=torch.bool)
        if inputs.context_valid_mask is None
        else inputs.context_valid_mask.detach().cpu().bool()
    )
    docs = (
        torch.zeros(n, s, dtype=torch.long)
        if inputs.document_ids is None else inputs.document_ids.detach().cpu()
    )
    anchor_docs = (
        torch.zeros(n, a, dtype=torch.long)
        if inputs.anchor_document_ids is None else inputs.anchor_document_ids.detach().cpu()
    )
    anchors = inputs.anchor_positions.detach().cpu()
    keep = inputs.block_keep_mask.detach().cpu().bool()
    if any(x.shape != (n, s) for x in (positions, valid, docs)):
        raise ValueError("context positions, validity and documents must be [N,S]")
    if anchor_docs.shape != (n, a) or keep.shape != (n, a):
        raise ValueError("anchor documents and validity must be [N,A]")
    if positions.is_floating_point() or anchors.is_floating_point():
        raise ValueError("context and anchor positions must be integers")
    if (anchors[keep] < 0).any():
        raise ValueError("anchor positions must be nonnegative")
    contexts, queries, writes, reads = [], [], [], []
    lengths, samples, boundaries = [], [], [0]
    slots = torch.arange(block_size)
    for sample in range(n):
        sequence_docs = docs[sample, valid[sample]]
        if sequence_docs.unique_consecutive().numel() != sequence_docs.unique().numel():
            raise ValueError("A document must occupy one contiguous context segment")
        for document in anchor_docs[sample, keep[sample]].unique().tolist():
            ai = torch.where(keep[sample] & (anchor_docs[sample] == document))[0]
            ap = anchors[sample, ai]
            ci = torch.where(valid[sample] & (docs[sample] == document))[0]
            cp = positions[sample, ci]
            if cp.numel() and ((cp < 0).any() or (cp[1:] <= cp[:-1]).any()):
                raise ValueError("Context positions must increase within each document")
            ci = ci[cp < ap.max()]
            cp = positions[sample, ci]
            qi = (sample * a * block_size + ai[:, None] * block_size + slots).flatten()
            keys = torch.cat((2 * cp + 1, (2 * ap[:, None]).expand(-1, block_size).flatten()))
            order = keys.argsort(stable=True)
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(order.numel())
            offset = boundaries[-1]
            contexts.append(sample * s + ci)
            queries.append(qi)
            writes.append(offset + inverse[:ci.numel()])
            reads.append(offset + inverse[ci.numel():])
            lengths.append(ci.numel())
            samples.append(sample)
            boundaries.append(offset + order.numel())

    def indices(parts):
        return (torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)).to(device)

    cpu_boundaries = torch.tensor(boundaries, dtype=torch.long)
    return GDNEventLayout(
        indices(contexts), indices(queries), indices(writes), indices(reads),
        cpu_boundaries.to(device), cpu_boundaries, tuple(lengths), tuple(samples),
        boundaries[-1],
    )
