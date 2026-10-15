from torch import nn


class Backbone(nn.Module):
    def forward(self, context):
        raise NotImplementedError
