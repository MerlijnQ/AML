import torch.nn as nn
import numpy as np
import math
from model.TCN import TCNModule
from model.bayesian import bayesianModule

class DHBCNN(nn.Module):
    def __init__(self, n_features, n_timesteps, dilation=2, k=3):
        super().__init__()

        n_channels = int(32 * np.sqrt(n_features))

        ins = (n_timesteps-1)/(2* (k-1)) + 1
        B = math.ceil(math.log2(ins))

        blocks = []
        blocks.append(TCNModule(dilation**0, n_channels, c_in = n_features, k=k))

        for b in range(1, B):
            dilation_b = dilation**b
            blocks.append(TCNModule(dilation_b, n_channels, k=k))

        self.blocks = nn.ModuleList(blocks)

        #Add global pooling layer to collapse timedimension
        self.pool = nn.AdaptiveAvgPool1d(1)
        #Size now becomes n_channels

        #Add bayesian linear layer
        self.mu = bayesianModule(n_channels)
        self.sigma = bayesianModule(n_channels)

    def feature_extractor(self, X):
        for block in self.blocks:
            X = block(X)
        X = self.pool(X)
        X = X.squeeze(-1)
        return X
        
    def forward(self, X):
        X = self.feature_extractor(X)
        mu = self.mu(X)
        sigma = self.sigma(X)
        return mu, sigma

    def predict_mu(self, X):
        self.feature_extractor(X)
        return self.mu(X)