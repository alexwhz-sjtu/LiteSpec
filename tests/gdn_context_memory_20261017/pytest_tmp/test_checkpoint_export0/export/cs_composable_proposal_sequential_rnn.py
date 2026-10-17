# Generated inference bundle from composable/proposal/sequential_rnn.py

"""DLite sequential replacement head; no unnecessary unary vocabulary projection."""
import torch
from .cs_dlite import DLiteSequentialHead
from .cs_composable_contracts import DraftOutput
from .cs_composable_proposal_parallel import ParallelProposal

class SequentialRNNProposal(ParallelProposal):

    def __init__(self, config):
        super().__init__(config)
        self.sequential_head = DLiteSequentialHead(head_type='rnn', vocab_size=config.vocab_size, sequential_rank=int(config.composable_config['proposal'].get('rank', 256)), hidden_size=config.hidden_size, max_prediction_length=config.block_size - 1)

    def forward(self, features, context, shared_ops):
        if context.inputs.mode == 'features':
            return super().forward(features, context, shared_ops)
        draft_hidden_states = self.prediction_hidden(features, context)
        if context.inputs.mode == 'sample':
            tokens, draft_logits = self.sequential_head.sample_block_tokens(hidden_states=draft_hidden_states.flatten(0, 1), first_prev_token_ids=context.inputs.anchor_token_ids.flatten(), temperature=context.inputs.temperature)
            draft_logits = draft_logits.reshape(*draft_hidden_states.shape[:-1], -1)
            return DraftOutput(draft_hidden_states, draft_logits=draft_logits, draft_token_ids=tokens.reshape(draft_hidden_states.shape[:-1]), draft_probabilities=torch.softmax(draft_logits.float() / (context.inputs.temperature or 1.0), -1), draft_state=features.draft_state)
        if context.inputs.previous_token_ids is None:
            raise ValueError('Teacher forcing requires previous_token_ids')
        latent = self.sequential_head.forward_teacher_forcing(hidden_states=draft_hidden_states, prev_token_ids=context.inputs.previous_token_ids)
        return DraftOutput(draft_hidden_states, draft_logits=self.sequential_head.project_logits(latent), draft_state=features.draft_state)
