import numpy as np
import torch
from einops.layers.torch import Rearrange
from torch import nn


class Aff(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones([1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, dim]))

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x


class MLP(nn.Module):

    def __init__(self, inp_chn, mid_chn, out_chn, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_chn, mid_chn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_chn, out_chn),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MLPBlock(nn.Module):

    def __init__(
        self,
        inp_chn,
        mid_chn,
        out_chn,
        dropout=0.0,
    ):
        super().__init__()
        self.affine = Aff(inp_chn)
        self.norm = nn.LayerNorm(inp_chn)
        self.mlp = nn.Sequential(
            MLP(inp_chn, mid_chn, out_chn, dropout),
        )

    def forward(self, x):
        x = self.affine(x)
        x = x + self.mlp(self.norm(x))
        return x


class ResMLP(nn.Module):

    def __init__(
        self,
        dim,
        hidden_dim,
        depth,
    ):
        super().__init__()

        self.mlp_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mlp_blocks.append(
                MLPBlock(
                    dim,
                    hidden_dim,
                    dim,
                )
            )

        self.affine = Aff(dim)

    def forward(self, x):

        for mlp_block in self.mlp_blocks:
            x = mlp_block(x)

        x = self.affine(x)

        return x
