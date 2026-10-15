from .markov import MarkovProposal
from .parallel import ParallelProposal
from .rnn import RNNProposal
from .selector import SelectorProposal
from .sequential_rnn import SequentialRNNProposal

PROPOSALS = {
    "parallel": ParallelProposal,
    "markov": MarkovProposal,
    "rnn": RNNProposal,
    "sequential_rnn": SequentialRNNProposal,
    "selector": SelectorProposal,
}


def build_proposal(config):
    return PROPOSALS[config.composable_config["proposal"]["type"]](config)
