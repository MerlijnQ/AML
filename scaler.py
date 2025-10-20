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