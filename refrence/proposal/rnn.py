# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: author_repository
# Repository: deepseek-ai/DeepSpec
# Revision: 005e03b81cec38b7da6399833d609ee89a2587f2
# Source: deepspec/modeling/dspark/markov_head.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/deepseek-ai__DeepSpec/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/markov_head.py
# License / upstream notices: refrence/licenses/dspark-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Preserve joint state/embedding/hidden projection, output branch, teacher prefix and state reset.


# BEGIN RNNHead — upstream lines 125-284
class RNNHead(VanillaMarkov):
    """RNN-based head that maintains recurrent state across positions within a block.

    Unlike the memoryless Markov heads, position k can access the full prefix
    history x_{<k} through a GRU-like recurrent state.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,
    ):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "rnn"
        self.hidden_size = hidden_size
        # Joint projection: [s_{k-1}; W1[x_{k-1}]; h_k] -> [gate; candidate; output]
        # W_g, W_c, W_o in R^{(2r+d) x r} parameterized as one linear layer.
        self.state_size = markov_rank
        self.joint_proj = nn.Linear(
            2 * markov_rank + hidden_size, 3 * markov_rank
        )

    def _rnn_step(
        self,
        state: torch.Tensor,
        prev_embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single RNN step.

        Args:
            state: [*, r] previous recurrent state
            prev_embeddings: [*, r] W1[x_{k-1}]
            hidden_states: [*, d] backbone hidden at step k

        Returns:
            new_state: [*, r]
            bias: [*, vocab_size]
        """
        z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
        proj = self.joint_proj(z)
        gate_raw, candidate_raw, output_raw = proj.chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        bias = self.project_bias(torch.tanh(output_raw))
        return new_state, bias

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Stateless single-step bias (state initialized to zero).

        This is used for compatibility but does NOT carry state across steps.
        For full RNN behavior, use apply_block_logits or sample_block_tokens.
        """
        assert hidden_states is not None
        prev_embeddings = self.get_prev_embeddings(token_ids)
        state = torch.zeros_like(prev_embeddings)
        _, bias = self._rnn_step(state, prev_embeddings, hidden_states)
        return bias

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply RNN bias during training (teacher-forced, unrolled over block_size).

        Args:
            base_logits: [B, num_blocks, block_size, V]
            token_ids: [B, num_blocks, block_size] - prev token ids for each position
            hidden_states: [B, num_blocks, block_size, d]
        """
        assert hidden_states is not None
        block_size = base_logits.size(-2)
        if block_size == 0:
            return base_logits

        leading_shape = base_logits.shape[:-2]  # e.g. [B, num_blocks]
        state = torch.zeros(
            *leading_shape,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )

        output_logits = []
        for k in range(block_size):
            prev_emb = self.get_prev_embeddings(token_ids[..., k])
            h_k = hidden_states[..., k, :]
            state, bias = self._rnn_step(state, prev_emb, h_k)
            output_logits.append(base_logits[..., k, :] + bias)

        return torch.stack(output_logits, dim=-2)

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive sampling with RNN state.

        Args:
            base_logits: [batch, proposal_len, vocab]
            first_prev_token_ids: [batch] - token preceding the first draft position
            hidden_states: [batch, proposal_len, d]
            temperature: sampling temperature

        Returns:
            sampled_tokens: [batch, proposal_len]
            corrected_logits: [batch, proposal_len, vocab]
        """
        assert hidden_states is not None
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
            return empty_tokens, base_logits

        state = torch.zeros(
            batch_size,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )

        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()

        for step_idx in range(proposal_len):
            prev_emb = self.get_prev_embeddings(prev_token_ids)
            h_k = hidden_states[:, step_idx, :]
            state, bias = self._rnn_step(state, prev_emb, h_k)

            step_logits = base_logits[:, step_idx, :] + bias
            corrected_logits.append(step_logits.unsqueeze(1))

            next_token_ids = sample_tokens(
                step_logits.unsqueeze(1),
                temperature=temperature,
            ).squeeze(1)
            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids

        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)

# END RNNHead
