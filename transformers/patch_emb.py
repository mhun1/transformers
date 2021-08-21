import torch

import numpy
import matplotlib.pyplot as plt

from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange, repeat


class PatchEmbedding(nn.Module):

    """Creates the patches for the model input"""

    def __init__(
        self,
        dim,
        channels=3,
        dropout_rate=0.1,
        image_height=256,
        image_width=256,
        image_depth=256,
        patch_height=16,
        patch_width=16,
        patch_depth=None,
    ):

        """

        :patch_height: Height of patches, default is 16. 
        :patch_width: Width of patches, default is 16.
        :patch_depth: Depth of patches, default is undefined. 
        """
        nn.Module.__init__(self)

        if not patch_depth:
            patch_dim = channels * patch_height * patch_width
            self._num_patches = (image_height // patch_height) * (
                image_width // patch_width
            )
            self._rearr = Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            )
            self._lin = nn.Linear(patch_dim, dim)
            self._pos = nn.Parameter(torch.randn(1, self._num_patches + 1, dim))
            self._token = nn.Parameter(torch.randn(1, 1, dim))

        else:
            patch_dim = channels * patch_height * patch_width * patch_depth
            self._num_patches = (
                (image_height // patch_height)
                * (image_width // patch_width)
                * (image_depth // patch_depth)
            )
            self._rearr = Rearrange(
                "b c (h p1) (w p2) (z p3) -> b (h w z) (p1 p2 p3 c)",
                p1=patch_height,
                p2=patch_width,
                p3=patch_depth,
            )
            self._lin = nn.Linear(patch_dim, dim)
            self._pos = nn.Parameter(torch.randn(1, self._num_patches + 1, dim))
            self._token = nn.Parameter(torch.randn(1, 1, dim))
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self._rearr(x)
        x = self._lin(x)
        b, _, _ = x.shape
        tokens = repeat(self._token, "() n d -> b n d", b=b)
        x = torch.cat((tokens, x), dim=1)
        x += self._pos[:, : (self._num_patches + 1)]
        x = self._dropout(x)
        return x
