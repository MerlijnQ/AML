import torch.nn as nn
import torch
import numpy as np
import math
from model.TCN import TCNModule
# from model.bayesian import bayesianModule
import torch.nn.functional as F

class DHBCNN(nn.Module):
    def __init__(self, n_features, n_timesteps, dilation=2, k=3):
        super().__init__()
        self.eps = 1e-6

        # n_channels = int(32 * np.sqrt(n_features))

        def devisor_by_8_closest(v):
            v_2 = max(8, ((v + 8/2) // 8) * 8 ) #Make sure it's multiple of 8 and at least 8. This way we are in line with common practices for channel sizes and it is friendly on hardware accelaration (e.g. bits are usually multiple of 8).
            if v_2 < 0.9 * v:
                v_2 += 8
            return int(v_2)

        def calculate_c0(n_features):
            width_m = 4 #Width multiplier
            # Heuristic to calculate initial channels based on number of features. We scale it by 4 to map to a reasonable n input c and make sure the count does not explode for high n_features
            v = int(round(width_m * math.sqrt(n_features)))
            return devisor_by_8_closest(v)
            
        
        c_0 = calculate_c0(n_features)
        c_prev = c_0

        ins = (n_timesteps-1)/(2* (k-1)) + 1
        B = math.ceil(math.log2(ins))

        blocks = []
        blocks.append(TCNModule(dilation**0, c_0, c_in = n_features, k=k))

        for b in range(1, B):
            g = 1.4 #Growth factor
            dilation_b = dilation**b
            c_x = c_0 * (g ** b)
            c_x = devisor_by_8_closest(c_x)
            c_x = min(256, c_x)  #Cap channels to 256 to avoid too large models
            blocks.append(TCNModule(dilation_b, c_x ,c_in=c_prev, k=k))
            c_prev = c_x

        self.blocks = nn.ModuleList(blocks)

        #Add global pooling layer to collapse timedimension
        self.pool = nn.AdaptiveAvgPool1d(1)
        #Size now becomes n_channels

        #Add bayesian linear layer
        # self.mu = bayesianModule(n_channels)
        # self.sigma = bayesianModule(n_channels)
        
        self.mu = nn.Linear(c_prev, 1)
        
        # nn.init.constant_(self.sigma.bias, 0.0)     # so elu(0) = 0
        # nn.init.normal_(self.sigma.weight, 0, 1e-4) #Initialize weights close to zero to prevent big initial sigma and potential instabilities

        # Initialize weights
        self._init_weights()
        
        #Initialize sigma to be 1: (softplus(math.log(math.e - 1)) + eps), or 0.5: softplus( math.log(0.5)) + eps
        self.sigma = nn.Linear(c_prev, 1)
        nn.init.constant_(self.sigma.bias, math.log(math.e - 1))  # ≈ 1 after softplus
        nn.init.normal_(self.sigma.weight, 0, 1e-4)  #Initialize weights close to zero to prevent big initial sigma and potential instabilities


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def sigma_head(self, X):
        X = self.sigma(X)
        #We need to following trick to prevent extreme values that we cannot use. I.e. it stabilizes. Derived from: https://arxiv.org/pdf/2012.14389
        # sigma = 1 + F.elu(X) + self.eps 
        # sigma = torch.clamp(sigma, min=self.eps)
        sigma = F.softplus(X) + self.eps
        return sigma
    
    def feature_extractor(self, X):
        for block in self.blocks:
            X = block(X)
        X = self.pool(X)
        X = X.squeeze(-1)
        return X
        
    def forward(self, X):
        X = self.feature_extractor(X)
        mu = self.mu(X)
        sigma = self.sigma_head(X)
        return mu, sigma

    # def predict_mu(self, X):
    #     self.feature_extractor(X)
    #     return self.mu(X)