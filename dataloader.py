import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random

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
        split_idx = (self._X.size(dim=0) - n_days_test)*self._day + self._input_window
        arg = [self._input_window, self._output_window, self._features]
        return (TimeSeriesDataset(self._orig_dataset[:split_idx], *arg), TimeSeriesDataset(self._orig_dataset[split_idx:], *arg))
    
    def train_val_split(self, n_days_train, n_days_val):
        if n_days_train > self._X.size(dim=0):
            raise ValueError(
                (
                    "The number of days wanted is outside the range of samples in the dataset (={})."
                ).format(len(self._data))
            )
        arg = [self._input_window, self._output_window, self._features]
        return (TimeSeriesDataset(self._orig_dataset[0:n_days_train*self._day+self._input_window], *arg), TimeSeriesDataset(self._orig_dataset[n_days_train*self._day:n_days_val*self._day+self._input_window], *arg))


    @property
    def data(self):
        return (self._X, self._y)
    
    @property
    def X(self):
        return self._X
    
    @property
    def y(self):
        return self._y
    
class ForwardTrainingCrossValidation():
    def __init__(self, dataset:TimeSeriesDataset, k_folds=10):
        self._dataset = dataset
        self._k_folds = k_folds
        remainder = len(dataset)%k_folds

        if remainder == 0:
            size_fold = int(len(dataset)/k_folds)
            self._split_idx = [(i+1)*size_fold for i in range(k_folds-1)]
        else:
            ls = [i for i in range(k_folds)]
            random_idx = random.sample(ls,remainder)

            size_fold = int(len(dataset)/k_folds)
            self._split_idx = []

            for i in range(k_folds):
                previous_split = self._split_idx[i-1] if i > 0 else 0
                self._split_idx.append(size_fold + previous_split)
                if i in random_idx:
                    self._split_idx[i] += 1

    def __getitem__(self, idx):
        return self._dataset.train_val_split(self._split_idx[idx],self._split_idx[idx+1])



if __name__ == "__main__":
    path_dataset = "dataset/continuous dataset.csv"

    dataset = pd.read_csv(path_dataset)

    # discard data from the month covid lockdown happened.
    dataset = dataset[dataset["datetime"] <= "2020-03-01 00:00:00"]

    input_window = 48
    output_window = 24
    features = ['nat_demand', 'T2M_toc']

    dat_time_series = TimeSeriesDataset(dataset, input_window, output_window, features)
    
    train, test= dat_time_series.train_test_split()
    ####################################################################################################################################
    ####################################################################################################################################

    dat = (dataset["nat_demand"])
    batch_size = 16
    n_folds = 10
    k_fold = ForwardTrainingCrossValidation(train, n_folds)
    for i in range(n_folds-2):
        training, validation = k_fold[i]

        train_loader = DataLoader(training, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)




