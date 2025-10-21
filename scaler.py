import torch

class MinMaxScaler:
    def __init__(self):
        self._min = None
        self._max = None

    def fit(self, data:torch.Tensor):
        self._min = data.min(axis=0).values
        self._max = data.max(axis=0).values

    def transform(self, data:torch.Tensor):
        return (data - self._min) / (self._max - self._min)
    
class ZScoreNormalization:
    def __init__(self):
        self._mean = None
        self._std = None

    def fit(self, data:torch.Tensor, n_discrete_features:int):
        data = data[:,:data.size(dim=1)-n_discrete_features]
        self._mean = data.mean(dim=0)
        self._std = data.std(dim=0)

    def transform(self, data:torch.Tensor, n_discrete_features:int):
        data_continuous = data[:,:,:data.size(dim=2)-n_discrete_features]
        data_discrete = data[:,:,data.size(dim=2)-n_discrete_features:]
        data_continuous = (data_continuous - self._mean) / self._std
        return torch.concat([data_continuous,data_discrete],dim=2)