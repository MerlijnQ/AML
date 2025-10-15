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
        features = ["nat_demand", "T2M_toc"] # should be discarded
        dataset = dataset[features]
        data = torch.tensor(dataset.values, dtype=torch.float32)
        self._data = data

        X, y = [],[]
        # get the x (features*input window) and y data points (1*output window) from dataset
        for i in range(int((len(self._data)-input_window)/self._day)):
            X_point = self._data[i*self._day:i*self._day+input_window,:]
            y_point = self._data[i*self._day+input_window:i*self._day+input_window+output_window,0]
            if len(y_point) == output_window: # ensure that y is length of output window 
                X.append(X_point)
                y.append(y_point)
            else:
                print("yes")
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
    
    def train_test_split(self, n_days_test=365):
        if n_days_test > self._X.size(dim=0):
            raise ValueError(
                (
                    "The number of days wanted is outside the range of samples in the dataset (={})."
                ).format(len(self._data))
            )
        split_idx = (self._X.size(dim=0) - n_days_test)*self._day
        arg = [self._input_window, self._output_window, self._features]
        return (TimeSeriesDataset(self._orig_dataset[:split_idx], *arg), TimeSeriesDataset(self._orig_dataset[split_idx:], *arg))


    @property
    def data(self):
        return (self._X, self._y)
    
    @property
    def X(self):
        return self._X
    
    @property
    def y(self):
        return self._y

if __name__ == "__main__":
    path_dataset = "dataset/continuous dataset.csv"

    dataset = pd.read_csv(path_dataset)
    # discard data from the month covid lockdown happened.
    dataset = dataset[dataset["datetime"] <= "2020-03-01 00:00:00"]

    features = ['nat_demand', 'T2M_toc']
    dat_time_series = TimeSeriesDataset(dataset, 48, 24, features)
    train, test= dat_time_series.train_test_split()
    print(len(train), len(test))
