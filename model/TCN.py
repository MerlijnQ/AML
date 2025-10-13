import torch.nn as nn
import torch.functional as F

class conv1D(nn.Module):
    def __init__(self, dilation, channels_in, channels_out, k):
        super().__init__()
        self.padding = (k-1)*dilation

        self.conv = nn.utils.weight_norm(
            nn.Conv1d(
            channels_in, channels_out, k, stride=1, dilation=dilation)
            )
        self.relu = nn.ReLU()

        #Optionally we add dropout for regularistion and generalisation during training.
        #Theoretically it is not strictly needed as we already do this with bayesian layer as well.
        self.dropout = nn.Dropout(0.2)  

    def forward(self, x):
        x = F.pad(x, (self.padding, 0)) 
        #Add padding to enforce causality (no peaking into the future)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TCNModule(nn.Module):
    def __init__(self, dilation, channels, c_in = None, k=3):
        super().__init__()

        channels_in = c_in if c_in is not None else channels
                    #c_in should be defined for the first layer (n of features)
        self.resid = (channels_in != channels) 
            #Need to project input to the right size if not already
        if self.resid:
            self.conv3 = nn.Conv1d(channels_in, channels, kernel_size=1)

        self.conv1 = conv1D(dilation, channels_in, channels, k)
        self.conv2 = conv1D(dilation, channels, channels, k)
        

    def forward(self, X):
        X_resid = X
        X = self.conv1(X)
        X = self.conv2(X)
        if self.resid:
            X_resid = self.conv3(X_resid)
        X = X + X_resid
        return X