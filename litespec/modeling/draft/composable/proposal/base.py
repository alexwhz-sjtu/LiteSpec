from torch import nn


class Proposal(nn.Module):
    def forward(self, features, context, shared_ops):
        raise NotImplementedError
