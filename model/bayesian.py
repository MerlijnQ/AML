import torch.nn as nn
import torch
import torch.nn.functional as F

class bayesianModule(nn.Module):
    def __init__(self, N_features:int)-> None:
        """A bayesian layer

        Args:
            N_features (int): Number of features as input to the layer.
        """

        super().__init__()
    
        self.W_mu = nn.Parameter(torch.zeros(1, N_features))
        self.W_sigma = nn.Parameter(torch.full((1, N_features), -5.0))

        self.b_mu = nn.Parameter(torch.zeros(1))
        self.b_sigma = nn.Parameter(torch.full((1, ), -5.0))

    def forward(self, X:torch.tensor) -> torch.tensor:
        """A forward pass through the bayesian layer.

        Args:
            X (torch.tensor): An input tensor.

        Returns:
            torch.tensor: An output tensor after transformations done in the bayesian layer.
        """

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

        out = F.linear(X, W, b)
        out = torch.clamp(out, -1e6, 1e6)
        return out