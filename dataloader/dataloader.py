from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
from dataloader.dataset import TimeSeriesDataset
from dataloader.scaler import ZScoreNormalization
from dataloader.one_hot_encode import one_hot_encode

DISCRETE_FEATURES = ['holiday', 'school', 'Holiday_ID1', 'Holiday_ID2', 'Holiday_ID3', 'Holiday_ID4', \
                        'Holiday_ID5', 'Holiday_ID6', 'Holiday_ID7', 'Holiday_ID8', 'Holiday_ID9', 'Holiday_ID10', \
                        'Holiday_ID11', 'Holiday_ID12', 'Holiday_ID13', 'Holiday_ID14', 'Holiday_ID15', 'Holiday_ID16', \
                        'Holiday_ID17', 'Holiday_ID18', 'Holiday_ID19', 'Holiday_ID20', 'Holiday_ID21', 'Holiday_ID22', \
                        'hour_sin', 'hour_cos', 'weekend']

class DataLoaderTimeSeries:
    def __init__(self, input_window: int = 24, output_window: int = 24, batch_size: int = 128) -> None: 
        """
        Initialize the dataloaders for the timeseries dataset.

        Args:
            input_window (int): input window size in hours.
            output_window (int): prediction in the future in hours.
            batch_size (int): size of the batch for the dataloaders.
        """
        self._input_window = input_window
        self._output_window = output_window
        self._batch_size = batch_size
        self.eps = 1e-8

        self._scaler = ZScoreNormalization()

        self._dataset = self._get_dataset()
        self._features = list(self._dataset.columns.values)
        self._preprocess_features()
        self._initialize_datasets()
        self._scale_input_features()
        self._scale_target_feature()
        self._update_loaders()

    def _get_dataset(self) -> pd.DataFrame:
        """
        Obtain the dataset from file.

        Returns:
            pd.DataFrame: The dataset with discarded data from 2020-03-01.
        """
        path_dataset = "dataset/continuous dataset.csv"
        dataset = pd.read_csv(path_dataset)

        # discard data from the month covid lockdown happened
        dataset = dataset[dataset["datetime"] <= "2020-03-01 00:00:00"]
        dataset["datetime"] = pd.to_datetime(dataset["datetime"])
        return dataset
    
    def _preprocess_features(self) -> None:
        """
        Preprocess features by one hot encoding and adding features.
        """
        # add one hot encoded Holiday ID feature
        df_one_hot_encoded = one_hot_encode("Holiday_ID", self._dataset["Holiday_ID"])
        self._dataset = pd.concat([self._dataset, df_one_hot_encoded], axis=1)

        # add cyclical feature hour and weekend to dataset
        hours = self._dataset["datetime"].dt.hour
        self._dataset["hour_sin"] = np.sin(2*np.pi*hours/24)
        self._dataset["hour_cos"] = np.cos(2*np.pi*hours/24)
        self._dataset["weekend"] = (self._dataset["datetime"].dt.dayofweek > 4).map({True: 1, False: 0})

        # update feature list
        self._features = list(self._dataset.columns.values)

        # remove excessive features
        self._features.remove("datetime")
        self._features.remove("Holiday_ID")
    
    def _initialize_datasets(self) -> None:
        """
        Initializes timeseries dataset instance and obtain train-val-test datasets.
        """
        dat_time_series = TimeSeriesDataset(self._dataset, self._input_window, self._output_window, self._features)
        train, self._test = dat_time_series.train_test_split()
        self._training, self._validation = train.train_test_split()

    def _scale_input_features(self) -> None:
        """
        Normalizes the input features.
        """
        n_discrete_features = len(set(self._features) & set(DISCRETE_FEATURES))

        self._scaler.fit(self._training.scalable_data,n_discrete_features)
        self._training.transform(self._scaler,n_discrete_features)
        self._validation.transform(self._scaler,n_discrete_features)
        self._test.transform(self._scaler,n_discrete_features)

    def _scale_target_feature(self) -> None:
        """
        Normalizes the the target feature.
        """
        y_train = self._training._y
        self.target_mean = y_train.mean().item()
        self.target_std = y_train.std().item()

        self._training._y = (self._training._y - self.target_mean) / (self.target_std + self.eps)
        self._validation._y = (self._validation._y - self.target_mean) / (self.target_std + self.eps)
        self._test._y = (self._test._y - self.target_mean) / (self.target_std + self.eps)

    def _inverse_transform_target(self, y_scaled:torch.Tensor) -> torch.Tensor:
        """
        Convert normalized target (y_scaled) back to original units.

        Args:
            y_scaled (torch.Tensor): normalized target.

        Returns:
            torch.Tensor: target in its original units.

        """
        return y_scaled * (self.target_std + self.eps) + self.target_mean
    
    def _update_loaders(self) -> None:
        """
        Initializes or updates the dataloaders for the train, validation and test set.
        """
        self._train_loader = DataLoader(self._training, batch_size=self._batch_size, shuffle=False)
        self._val_loader = DataLoader(self._validation, batch_size=self._batch_size, shuffle=False)
        self._test_loader = DataLoader(self._test, batch_size=self._batch_size, shuffle=False)

    def remove_feature(self, feature:str) -> None:
        """
        Removes a feature from the datasets and updates the dataloaders.

        Args:
            idx (int): index of the feature to be removed.
        """
        self._training.remove_feature(feature)
        self._validation.remove_feature(feature)
        self._test.remove_feature(feature)
        self._features.remove(feature)
        self._update_loaders()

    def get_feature_at_index(self, idx: int) -> str:
        
        """
        Returns the feature at the given index.

        Args:
            idx (int): the index of the feature.

        Returns:
            str: the feature at index.
        """
        return self._features[idx]

    @property
    def features(self) -> list[str]:
        """
        Returns the features.

        Returns:
            list[str]: the features of the dataset.
        """
        return self._features

    @property
    def train_loader(self) -> DataLoader:
        """
        Returns the train dataloader.

        Returns:
            DataLoader: the train dataloader.
        """
        return self._train_loader
    
    @property
    def validation_loader(self) -> DataLoader:
        """
        Returns the validation dataloader.

        Returns:
            DataLoader: the validation dataloader.
        """
        return self._val_loader
    
    @property
    def test_loader(self) -> DataLoader:
        """
        Returns the test dataloader.

        Returns:
            DataLoader: the test dataloader.
        """
        return self._test_loader
