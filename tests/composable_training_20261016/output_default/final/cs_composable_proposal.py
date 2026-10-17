# Generated inference bundle from composable/proposal/__init__.py

from .cs_composable_proposal_markov import MarkovProposal
from .cs_composable_proposal_parallel import ParallelProposal
from .cs_composable_proposal_rnn import RNNProposal
from .cs_composable_proposal_selector import SelectorProposal
from .cs_composable_proposal_sequential_rnn import SequentialRNNProposal
PROPOSALS = {'parallel': ParallelProposal, 'markov': MarkovProposal, 'rnn': RNNProposal, 'sequential_rnn': SequentialRNNProposal, 'selector': SelectorProposal}

def build_proposal(config):
    return PROPOSALS[config.composable_config['proposal']['type']](config)
