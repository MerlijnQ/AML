import torch

class ZScoreNormalization:
    def __init__(self):
        """Z score normalization using the mean and standard deviation."""
        self._mean = None
        self._std = None

    def fit(self, data:torch.Tensor, n_discrete_features:int):
        """Compute the mean and std from the continuous data."""
        data = data[:,:data.size(dim=1)-n_discrete_features]
        self._mean = data.mean(dim=0)
        self._std = data.std(dim=0)

    def transform(self, data:torch.Tensor, n_discrete_features:int):
        """Transforms the continuous data."""
        data_continuous = data[:,:,:data.size(dim=2)-n_discrete_features]
        data_discrete = data[:,:,data.size(dim=2)-n_discrete_features:]
        data_continuous = (data_continuous - self._mean) / self._std
        return torch.concat([data_continuous,data_discrete],dim=2)