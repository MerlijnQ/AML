import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

FEATURES = ['nat_demand', 'T2M_toc', 'QV2M_toc', 'TQL_toc', 'W2M_toc',
       'T2M_san', 'QV2M_san', 'TQL_san', 'W2M_san', 'T2M_dav', 'QV2M_dav',
       'TQL_dav', 'W2M_dav', 'Holiday_ID', 'holiday', 'school']

class TimeSeriesDataset(Dataset):
    def __init__(self, dataset, input_window: {24,48,72}, output_window = 24, features=FEATURES):
        self._orig_dataset = dataset
        self._input_window = input_window
        self._output_window = output_window
        self._features = features
        self._day = 24

        dataset = dataset.sort_values("datetime", ascending=True)
        dataset = dataset[features]
        self._data = torch.tensor(dataset.values, dtype=torch.float32)

        X, y = [],[]
        # get the x (features*input window) and y data points (1*1) from dataset
        for i in range(int((len(self._data)-input_window-output_window))):
            X_point = self._data[i:i+input_window,:]
            y_point = self._data[i+input_window+output_window-1,0]
            X.append(X_point)
            y.append(y_point)
        self._X = torch.stack(X,dim=0)
        self._y = torch.stack(y,dim=0)

    def __len__(self):
        return self._X.size(dim=0)

    def __getitem__(self, index):
        if index > self._X.size(dim=0):
            raise ValueError(
                (
                    "The index is outside the range of the dataset (={})."
                ).format(len(self._data))
            )
        return self._X[index], self._y[index]
    
    def train_test_split(self, n_years=1):
        if n_years > self._X.size(dim=0):
            raise ValueError(
                (
                    "The number of days wanted is outside the range of samples in the dataset (={})."
                ).format(len(self._data))
            )
        time = self._orig_dataset["datetime"].iloc[-1]
        split_idx_train = time - pd.DateOffset(years=n_years)
        split_idx_test = split_idx_train - pd.DateOffset(hours=self._input_window+self._output_window)

        dat_train = self._orig_dataset[self._orig_dataset["datetime"]<=split_idx_train]
        dat_test = self._orig_dataset[self._orig_dataset["datetime"]>split_idx_test]

        arg = [self._input_window, self._output_window, self._features]
        return (TimeSeriesDataset(dat_train, *arg), TimeSeriesDataset(dat_test, *arg))

    @property
    def data(self):
        return (self._X, self._y)
    
    @property
    def X(self):
        return self._X
    
    @property
    def y(self):
        return self._y

