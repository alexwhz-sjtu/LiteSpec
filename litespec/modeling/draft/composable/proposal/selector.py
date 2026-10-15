"""DFlash2 strict unary top-K, predecessor-conditioned categorical selector."""

import torch

from ...dflash2 import CandidateSelector
from ..contracts import DraftOutput
from .parallel import ParallelProposal


class SelectorProposal(ParallelProposal):
    def __init__(self, config):
        super().__init__(config)
        recipe = config.composable_config["proposal"]
        self.candidate_selector = CandidateSelector(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            state_rank=int(recipe.get("rank", 256)),
            top_k=int(recipe.get("top_k", 16)),
            initializer_range=config.initializer_range,
        )
        self.stop_gradient = bool(recipe.get("stop_gradient", False))

    def forward(self, features, context, shared_ops):
        if context.inputs.mode == "features":
            return super().forward(features, context, shared_ops)
        draft_hidden_states = self.prediction_hidden(features, context)
        base_logits = shared_ops.lm_head(draft_hidden_states)
        output = DraftOutput(
            draft_hidden_states,
            base_logits=base_logits,
            draft_state=features.draft_state,
        )
        candidate_logits, candidate_ids = base_logits.topk(
            self.candidate_selector.top_k, dim=-1
        )
        if context.inputs.mode == "sample":
            tokens, draft_probabilities = self.candidate_selector.select_path(
                candidate_ids=candidate_ids.flatten(0, 1),
                unary_logits=candidate_logits.flatten(0, 1),
                hidden_states=draft_hidden_states.flatten(0, 1),
                anchor_token_ids=context.inputs.anchor_token_ids.flatten(),
                temperature=context.inputs.temperature,
            )
            output.draft_token_ids = tokens.reshape(draft_hidden_states.shape[:-1])
            if draft_probabilities is None:
                # Recover the actual rows along the greedy path too.
                predecessors = torch.cat(
                    (
                        context.inputs.anchor_token_ids[..., None],
                        output.draft_token_ids[..., :-1],
                    ),
                    -1,
                )
                scores = self.candidate_selector.score_candidates(
                    candidate_ids=candidate_ids,
                    unary_logits=candidate_logits,
                    hidden_states=draft_hidden_states,
                    predecessor_ids=predecessors,
                )
                draft_probabilities = scores.float().softmax(-1)
            else:
                draft_probabilities = draft_probabilities.reshape_as(candidate_logits)
            output.draft_probabilities = base_logits.new_zeros(
                base_logits.shape, dtype=torch.float32
            ).scatter(-1, candidate_ids, draft_probabilities)
        else:
            if context.inputs.previous_token_ids is None:
                raise ValueError("Selector scoring requires previous_token_ids")
            output.candidate_logits = self.candidate_selector.score_candidates(
                candidate_ids=candidate_ids,
                unary_logits=candidate_logits.detach()
                if self.stop_gradient
                else candidate_logits,
                hidden_states=draft_hidden_states.detach()
                if self.stop_gradient
                else draft_hidden_states,
                predecessor_ids=context.inputs.previous_token_ids,
            )
        if context.inputs.mode == "sample":
            output.draft_logits = output.draft_probabilities.log() * (
                context.inputs.temperature or 1.0
            )
        output.candidate_ids = candidate_ids
        return output
