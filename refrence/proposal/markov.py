# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: author_repository
# Repository: deepseek-ai/DeepSpec
# Revision: 005e03b81cec38b7da6399833d609ee89a2587f2
# Source: deepspec/modeling/dspark/markov_head.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/deepseek-ai__DeepSpec/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/markov_head.py
# License / upstream notices: refrence/licenses/dspark-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Additive full-vocabulary correction and sequential proposal conditioning.


# BEGIN VanillaMarkov — upstream lines 8-90
class VanillaMarkov(nn.Module):
    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = "vanilla"
        assert self.markov_rank > 0, (
            f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}."
        )
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        del hidden_states
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return logits + self.compute_step_bias(token_ids, hidden_states)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if base_logits.size(2) == 0:
            return base_logits
        markov_bias = self.compute_step_bias(token_ids, hidden_states)
        return base_logits + markov_bias

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
            return empty_tokens, base_logits

        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()
        for step_idx in range(proposal_len):
            step_hidden = None if hidden_states is None else hidden_states[:, step_idx, ...]
            step_logits = self.apply_step_logits(
                base_logits[:, step_idx, :],
                token_ids=prev_token_ids,
                hidden_states=step_hidden,
            )
            corrected_logits.append(step_logits.unsqueeze(1))
            next_token_ids = sample_tokens(
                step_logits.unsqueeze(1),
                temperature=temperature,
            ).squeeze(1)
            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids
        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)

# END VanillaMarkov
