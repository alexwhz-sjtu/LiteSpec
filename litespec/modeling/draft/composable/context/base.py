from torch import nn


class Context(nn.Module):
    """Inject target features inside the trainable/FSDP forward."""

    def forward(self, inputs):
        raise NotImplementedError
