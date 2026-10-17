# Generated inference bundle from composable/proposal/rnn.py

from .cs_dspark import RNNMarkovHead
from .cs_composable_proposal_markov import MarkovProposal

class RNNProposal(MarkovProposal):
    head_class = RNNMarkovHead
