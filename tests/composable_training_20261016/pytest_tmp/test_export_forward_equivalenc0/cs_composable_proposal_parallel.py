# Generated inference bundle from composable/proposal/parallel.py

import torch
from .cs_composable_contracts import DraftOutput
from .cs_composable_proposal_base import Proposal

class ParallelProposal(Proposal):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def prediction_hidden(self, features, context):
        batch_size, num_anchors = context.inputs.anchor_positions.shape
        return features.draft_hidden_states.reshape(batch_size, num_anchors, self.config.block_size, -1)[..., 1:, :]

    def forward(self, features, context, shared_ops):
        draft_hidden_states = self.prediction_hidden(features, context)
        output = DraftOutput(draft_hidden_states=draft_hidden_states, draft_state=features.draft_state)
        if context.inputs.mode == 'features':
            return output
        if shared_ops is None:
            raise ValueError('Scoring requires the frozen target SharedOps')
        draft_logits = shared_ops.lm_head(draft_hidden_states)
        output.draft_logits = output.base_logits = draft_logits
        if context.inputs.mode == 'sample':
            temperature = context.inputs.temperature
            draft_probabilities = torch.softmax(draft_logits.float() / (temperature or 1.0), -1)
            output.draft_token_ids = draft_logits.argmax(-1) if temperature == 0 else torch.multinomial(draft_probabilities.flatten(0, -2), 1).reshape(draft_logits.shape[:-1])
            output.draft_probabilities = draft_probabilities
        return output
