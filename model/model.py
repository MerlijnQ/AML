import torch.nn as nn
import numpy as np
import math
from model.TCN import TCNModule

class model(nn.Module):
    def __init__(self, n_features, n_timesteps, dilation=2, k=3):
        super().__init__()

        n_channels = 32 * np.sqrt(n_features)

        ins = (n_timesteps-1)/(2* (k-1)) + 1
        B = math.ceil(math.log2(ins))

        blocks = []
        blocks.append(TCNModule(dilation, n_channels, c_in = n_features, k=k))

        for _ in range(0, B-1):
            blocks.append(TCNModule(dilation, n_channels, k=k))

        self.blocks = nn.ModuleList(blocks)

        #Add global pooling layer to collapse timedimension
        self.pool = nn.AvgPool1d(kernel_size=1)
        #Size now becomes n_channels

        #Add bayesian linear layer

    def forward(self, X):
        for block in self.blocks:
            X = block(X)
        X = self.pool(X)
        #add beysian linear layer here
        #Add 2 output heads.
