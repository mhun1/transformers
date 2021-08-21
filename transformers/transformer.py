import torch

from torch import nn
from norm import Norm
from attention import MHSA
from mlp import MLP


class Transformer(nn.Module):

    """Implementation of the _transformer block"""

    def __init__(self, dim, depth, heads, mlp_dim, dropout_rate):
        """TODO: to be defined.

        :dim: TODO
        :depth: TODO
        :heads: TODO
        :dim_heads: TODO
        :mlp: TODO

        """
        nn.Module.__init__(self)

        self._layers = nn.ModuleList()
        for _ in range(depth):
            self._layers.append(nn.ModuleList([Norm(dim), MHSA(dim, heads=heads), Norm(dim), MLP(dim,mlp_dim,dropout_rate)]))

    def forward(self, x):
        for n_0, mhsa, n_1, mlp in self._layers:
            x = mhsa(n_0(x)) + x
            x = mlp(n_1(x)) + x
            return x
