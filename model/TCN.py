import torch.nn as nn
import torch.nn.functional as F
import torch

class conv1D(nn.Module):
    def __init__(self, dilation:int, channels_in:int, channels_out:int, k:int)->None:

        """A custom convolutional layer sub-block with padding, weightnorm, ReLu activation and dropout.

        Args:
            dilation (int): The dilation in the convolutional layer.
            channels_in (int): The number of channels fed into the convolutional layer.
            channels_out (int): The number of channels to which the convolutional layer transforms the input.
            k (int): The kernel size.
        """

        super().__init__()
        self.padding = (k-1)*dilation

        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
            channels_in, channels_out, k, stride=1, dilation=dilation)
            )
        self.relu = nn.ReLU()

        #Optionally we add dropout for regularistion and generalisation during training.
        #Theoretically it is not strictly needed as we already do this with bayesian layer as well.
        self.dropout = nn.Dropout(0.2)  

    def forward(self, x:torch.tensor)->torch.tensor:

        """A forward pass through the convolution layer sub-block

        Args:
            x (torch.tensor): Input data

        Returns:
            torch.tensor: Output data
        """
        x = F.pad(x, (self.padding, 0)) 
        #Add padding to enforce causality (no peaking into the future)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TCNModule(nn.Module):
    def __init__(self, dilation:int, channels:int, c_in:int|None = None, k:int=3)-> None:

        """A TCN block

        Args:
            dilation (int): Dilation size used in the convolutional layers.
            channels (int): Number of output channels
            c_in (int | None, optional): Number of input channels if they differ from the number of output channels.
            Defaults to None.
            k (int, optional): Kernel size. Defaults to 3.
        """

        super().__init__()

        channels_in = c_in if c_in is not None else channels
                    #c_in should be defined for the first layer (n of features)
        self.resid = (channels_in != channels) 
            #Need to project input to the right size if not already
        if self.resid:
            self.conv3 = nn.Conv1d(channels_in, channels, kernel_size=1)

        self.conv1 = conv1D(dilation, channels_in, channels, k)
        self.conv2 = conv1D(dilation, channels, channels, k)
        

    def forward(self, X:torch.tensor)->torch.tensor:

        """A forward pass through the TCN block with residual connection.

        Args:
            X (torch.tensor): Input tensor

        Returns:
            torch.tensor: Output tensor
        """

        X_resid = X
        X = self.conv1(X)
        X = self.conv2(X)
        if self.resid:
            X_resid = self.conv3(X_resid)
        X = X + X_resid
        return F.relu(X)