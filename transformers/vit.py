import torch

from torch import nn
from patch_emb import PatchEmbedding
from transformer import Transformer

class ViT(nn.Module):

    """Implementation of the Vision Transformer presented in the paper.
        -> link to paper
    """

    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        num_classes,
        channels=3,
        drop_patch=0.1,
        drop_mlp=0.1,
        image_sizes = [256,256],
        patch_sizes = [16,16]
    ):
        
        """TODO: to be defined. """
        nn.Module.__init__(self)
        assert len(image_sizes) == len(patch_sizes), "Image size and patch size must have the same length."
        
        img_z, p_z = None, None
        if len(image_sizes) == 2:
            img_h, img_w = image_sizes
            p_h, p_w = patch_sizes 
        else:
            img_h, img_w, img_z = image_sizes
            p_h, p_w, p_z = patch_sizes 

        self._patch_emb = PatchEmbedding(
            dim,
            channels=channels,
            dropout_rate=drop_patch,
            image_height=img_h,
            image_width=img_w,
            image_depth=img_z,
            patch_height=p_h,
            patch_width=p_w,
            patch_depth=p_z,
        )

        self._transformer = Transformer(dim, depth, heads, mlp_dim, drop_mlp)
        self._mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        x = self._patch_emb(x)
        x = self._transformer(x).mean(dim=1)
        return self._mlp_head(x)

x = torch.zeros([1,3,256,256,96])
model = ViT(1024,5,8,1024,1000, image_sizes=[256,256,96], patch_sizes=[16,16,16])
print(model(x).shape)
