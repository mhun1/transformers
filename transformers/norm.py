import torch

from torch import nn


class Norm(nn.Module):

    """Implementation of the pre-norm as described in the paper."""

    def __init__(self, dim):
        """TODO: to be defined.

        :dim: TODO

        """
        nn.Module.__init__(self)
        self._norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self._norm(x)
