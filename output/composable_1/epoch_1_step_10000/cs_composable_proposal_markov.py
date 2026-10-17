# Generated inference bundle from composable/proposal/markov.py

"""DSpark additive logits; original head math reused from LiteSpec."""
from .cs_dspark import VanillaMarkovHead
from .cs_composable_contracts import DraftOutput
from .cs_composable_proposal_parallel import ParallelProposal

class MarkovProposal(ParallelProposal):
    head_class = VanillaMarkovHead

    def __init__(self, config):
        super().__init__(config)
        recipe = config.composable_config['proposal']
        kwargs = dict(vocab_size=config.vocab_size, markov_rank=int(recipe.get('rank', 256)))
        if self.head_class is not VanillaMarkovHead:
            kwargs['hidden_size'] = config.hidden_size
        self.markov_head = self.head_class(**kwargs)

    def forward(self, features, context, shared_ops):
        if context.inputs.mode == 'features':
            return super().forward(features, context, shared_ops)
        draft_hidden_states = self.prediction_hidden(features, context)
        base_logits = shared_ops.lm_head(draft_hidden_states)
        if context.inputs.mode == 'sample':
            import torch
            shape = draft_hidden_states.shape
            tokens, draft_logits = self.markov_head.sample_block_tokens(base_logits.flatten(0, 1), hidden_states=draft_hidden_states.flatten(0, 1), first_prev_token_ids=context.inputs.anchor_token_ids.flatten(), temperature=context.inputs.temperature)
            draft_logits = draft_logits.reshape(*shape[:-1], -1)
            return DraftOutput(draft_hidden_states, draft_logits, base_logits, tokens.reshape(shape[:-1]), torch.softmax(draft_logits.float() / (context.inputs.temperature or 1.0), -1), draft_state=features.draft_state)
        if context.inputs.previous_token_ids is None:
            raise ValueError('Teacher forcing requires previous_token_ids')
        draft_logits = self.markov_head.apply_block_logits(base_logits, token_ids=context.inputs.previous_token_ids, hidden_states=draft_hidden_states)
        return DraftOutput(draft_hidden_states, draft_logits, base_logits, draft_state=features.draft_state)
