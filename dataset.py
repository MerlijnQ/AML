import torch
from torch.utils.data import Dataset
import pandas as pd
from scaler import MinMaxScaler

FEATURES = ['nat_demand', 'T2M_toc', 'QV2M_toc', 'TQL_toc', 'W2M_toc',
       'T2M_san', 'QV2M_san', 'TQL_san', 'W2M_san', 'T2M_dav', 'QV2M_dav',
       'TQL_dav', 'W2M_dav', 'Holiday_ID', 'holiday', 'school']

class TimeSeriesDataset(Dataset):
    def __init__(self, dataset, input_window: {24,48,72}, output_window = 24, features:list=FEATURES):
        self._orig_dataset = dataset
        self._input_window = input_window
        self._output_window = output_window
        self._features = features

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
        self._scaled_X = None

    def __len__(self):
        return self._X.size(dim=0)

    def __getitem__(self, index):
        if index > self._X.size(dim=0):
            raise ValueError(
                (
                    "The index is outside the range of the dataset (={})."
                ).format(len(self._data))
            )

        # permute(1, 0) changes x from [T, C] to [C, T]
        if self._scaled_X == None:
            return self._X[index].permute(1, 0), self._y[index]
        else:
            return self._scaled_X[index].permute(1, 0), self._y[index]
    
    def train_test_split(self, n_years=1):
        print("train_test_split")
        time = self._orig_dataset["datetime"].iloc[-1]
        split_idx_train = time - pd.DateOffset(years=n_years)
        split_idx_test = split_idx_train - pd.DateOffset(hours=self._input_window+self._output_window)

        dat_train = self._orig_dataset[self._orig_dataset["datetime"]<=split_idx_train]
        dat_test = self._orig_dataset[self._orig_dataset["datetime"]>split_idx_test]

        arg_train = [self._input_window, self._output_window, self._features.copy()]
        arg_test = [self._input_window, self._output_window, self._features.copy()]
        return (TimeSeriesDataset(dat_train, *arg_train), TimeSeriesDataset(dat_test, *arg_test))

    def transform(self, scaler: MinMaxScaler):
        self._scaled_X = scaler.transform(self._X)

    def remove_feature(self, feature):
        feature_index = self._features.index(feature)
        flattened_X = self._X.view(-1,len(self._features))
        X_feature_removed = torch.cat((flattened_X[:,:feature_index], flattened_X[:,feature_index+1:]), dim=1)

        if self._scaled_X is not None:
            flattened_scaled_X = self._scaled_X.view(-1,len(self._features))
            scaled_X_feature_removed = torch.cat((flattened_scaled_X[:,:feature_index], flattened_scaled_X[:,feature_index+1:]), dim=1)

        self._features.remove(feature)
        self._X = X_feature_removed.view(len(self._y), self._input_window, len(self._features))

        if self._scaled_X is not None:
            self._scaled_X = scaled_X_feature_removed.view(len(self._y), self._input_window, len(self._features))

    @property
    def data(self):
        return (self._X, self._y) if self._scaled_X == None else (self._scaled_X, self._y)
    
    @property
    def X(self):
        return self._X
    
    @property
    def y(self):
        return self._y
