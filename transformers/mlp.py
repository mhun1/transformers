import torch

from torch import nn


class MLP(nn.Module):

    """Docstring for MLP. """

    def __init__(self, dim, hidden, dropout_rate):
        """TODO: to be defined.

        :dim: TODO
        :hidden: TODO
        :dropout_rate: TODO

        """
        nn.Module.__init__(self)

        self._dim = dim
        self._hidden = hidden
        self._dropout_rate = dropout_rate
        self._mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self._mlp(x)
