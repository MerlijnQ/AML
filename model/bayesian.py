import torch.nn as nn
import torch
import torch.nn.functional as F

class bayesianModule(nn.Module):
    def __init__(self, N_features):
        super().__init__()
    
        self.W_mu = nn.Parameter(torch.zeros(N_features, 1))
        self.W_sigma = nn.Parameter(torch.full((N_features, 1), -0.5))

        self.b_mu = nn.Parameter(torch.zeros(1))
        self.b_sigma = nn.Parameter(torch.full((1, ), -0.5))

    def forward(self, X):
        # ensure sigma is always positive (as STD is by definition > 0).
        # We ensure this through softplus
        W_sigma = torch.log1p(torch.exp(self.W_sigma))
        b_sigma = torch.log1p(torch.exp(self.b_sigma))

        #Returns a tensor with the same size as W_sigma or b_sigma drawn from
        #a normal distribution
        eps_w = torch.randn_like(W_sigma)
        eps_b = torch.randn_like(b_sigma)

        #Reparameterization trick. This is too avoid breaking gradients. 
        #We are basically sampling noise in a smart way
        W = self.W_mu + W_sigma * eps_w
        b = self.b_mu + b_sigma * eps_b

        return F.linear(X, W, b)