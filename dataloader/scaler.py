import torch

class ZScoreNormalization:
    
    def __init__(self) -> None:
        """
        Initializes the Z score normalization class with mean and standard deviation variables.
        """
        self._mean = None
        self._std = None

    def fit(self, data:torch.Tensor, n_discrete_features:int) -> None:
        """
        Compute the mean and std from the continuous data.

        Args:
            data (torch.Tensor): the data X, including continuous (and discrete) variables.
            n_discrete_features (int): the number of discrete features.
        """
        data = data[:,:data.size(dim=1)-n_discrete_features]
        self._mean = data.mean(dim=0)
        self._std = data.std(dim=0)

    def transform(self, data:torch.Tensor, n_discrete_features:int) -> torch.Tensor:
        """
        Transforms the continuous data.

        Args:
            data (torch.Tensor): the data X, including continuous (and discrete) variables.
            n_discrete_features (int): the number of discrete features.

        Returns:
            torch.Tensor: the normalized continuous data concatenated with data of the discrete features.
        """
        data_continuous = data[:,:,:data.size(dim=2)-n_discrete_features]
        data_discrete = data[:,:,data.size(dim=2)-n_discrete_features:]
        data_continuous = (data_continuous - self._mean) / self._std
        return torch.concat([data_continuous,data_discrete],dim=2)