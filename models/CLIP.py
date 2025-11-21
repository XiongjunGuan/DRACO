"""
Description:
Author: Xiongjun Guan
Date: 2025-01-15 10:14:52
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2025-01-18 01:33:08

Copyright (C) 2025 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ProjectionHead(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, projection_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
        )

    def forward(self, x):
        x = self.projection(x)
        return x
