# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: author_repository
# Repository: z-lab/dflash
# Revision: 07ebd93db9f472af339b644bb70221ad8428328a
# Source: dflash/model.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/z-lab__dflash/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py
# License / upstream notices: refrence/licenses/dflash-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Top-K support with predecessor-conditioned walk, not global Viterbi.


# BEGIN CandidateSelector — upstream lines 515-547
class CandidateSelector(nn.Module):
    def __init__(self, config):
        super().__init__()
        rank = int(_draft_value(config, "selector_rank"))
        self.top_k = int(_draft_value(config, "selector_top_k"))
        self.predecessor_codebook = nn.Embedding(config.vocab_size, rank)
        self.successor_codebook = nn.Embedding(config.vocab_size, rank)
        self.hidden_projection = nn.Linear(config.hidden_size, rank, bias=False)

    def select(self, hidden, logits, anchor_ids, temperature):
        unary, candidates = torch.topk(logits, self.top_k, dim=-1, sorted=False)
        hidden = self.hidden_projection(hidden)
        predecessor = anchor_ids
        path, q_rows = [], []
        for position in range(hidden.shape[1]):
            scores = unary[:, position] + torch.einsum(
                "br,bkr->bk",
                self.predecessor_codebook(predecessor) * hidden[:, position],
                self.successor_codebook(candidates[:, position]),
            )
            if temperature > 0:
                q = _sampling_probs(scores[:, None], temperature)[:, 0]
                index = _sample_probs(q)
                q_rows.append(q)
            else:
                index = torch.argmax(scores, dim=-1)
            predecessor = candidates[:, position].gather(-1, index[:, None])[:, 0]
            path.append(predecessor)
        return (
            torch.stack(path, dim=1),
            candidates,
            torch.stack(q_rows, dim=1) if q_rows else None,
        )

# END CandidateSelector
