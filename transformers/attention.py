import torch
import math
from torch import nn

x = torch.ones([1, 5, 36])


class MHSA(nn.Module):

    """Implementation of the Multi-headed self-attention mechanism."""

    def __init__(self, k, heads=8):
        """

        :k: The input dimension
        :heads: Amount of heads to use.  

        """
        nn.Module.__init__(self)

        self._k = k
        self._h = heads
        self._keys = nn.Linear(k, k * heads, bias=False)
        self._queries = nn.Linear(k, k * heads, bias=False)
        self._values = nn.Linear(k, k * heads, bias=False)
        self._heads = nn.Linear(heads * k, k)

    def forward(self, x):
        b, c, k = x.size()

        keys = self._keys(x).view(b, c, self._h, k)

        queries = self._queries(x).view(b, c, self._h, k)
        values = self._values(x).view(b, c, self._h, k)
        dot = torch.einsum("bchk,bihe->bhci", queries, keys) / math.sqrt(k)
        dot = torch.softmax(dot, dim=-1)
        out = torch.einsum("bchd,bdce->bhce", dot, values)
        out = torch.einsum("bthe,khe->btk", out, self._heads.weight.view(k, self._h, k))
        return out + self._heads.bias
