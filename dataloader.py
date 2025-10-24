from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from dataset import TimeSeriesDataset
from scaler import ZScoreNormalization
from one_hot_encode import one_hot_encode

DISCRETE_FEATURES = ['holiday', 'school', 'Holiday_ID1', 'Holiday_ID2', 'Holiday_ID3', 'Holiday_ID4', \
                        'Holiday_ID5', 'Holiday_ID6', 'Holiday_ID7', 'Holiday_ID8', 'Holiday_ID9', 'Holiday_ID10', \
                        'Holiday_ID11', 'Holiday_ID12', 'Holiday_ID13', 'Holiday_ID14', 'Holiday_ID15', 'Holiday_ID16', \
                        'Holiday_ID17', 'Holiday_ID18', 'Holiday_ID19', 'Holiday_ID20', 'Holiday_ID21', 'Holiday_ID22', \
                        'hour_sin', 'hour_cos', 'weekend']

class DataLoaderTimeSeries:
    def __init__(self, input_window:{24,48,72}, output_window = 24, batch_size = 128):
        self._input_window = input_window
        self._output_window = output_window
        self._batch_size = batch_size
        self.eps = 1e-8

        self._scaler = ZScoreNormalization()

        self._dataset = self._get_dataset()
        self._features = list(self._dataset.columns.values)

        df_one_hot_encoded = one_hot_encode("Holiday_ID", self._dataset["Holiday_ID"])
        self._dataset = pd.concat([self._dataset, df_one_hot_encoded], axis=1)
        hours = self._dataset["datetime"].dt.hour
        self._dataset["hour_sin"] = np.sin(2*np.pi*hours/24)
        self._dataset["hour_cos"] = np.cos(2*np.pi*hours/24)
        self._dataset["weekend"] = (self._dataset["datetime"].dt.dayofweek > 4).map({True: 1, False: 0})

        self._features = list(self._dataset.columns.values)
        self._features.remove("datetime")
        self._features.remove("Holiday_ID")

        self._initialize_dataset()

    def _get_dataset(self):
        path_dataset = "dataset/continuous dataset.csv"
        dataset = pd.read_csv(path_dataset)

        # discard data from the month covid lockdown happened.
        dataset = dataset[dataset["datetime"] <= "2020-03-01 00:00:00"]
        dataset["datetime"] = pd.to_datetime(dataset["datetime"])
        return dataset

    
    def _initialize_dataset(self):
        dat_time_series = TimeSeriesDataset(self._dataset, self._input_window, self._output_window, self._features)
        train, self._test = dat_time_series.train_test_split()
        self._training, self._validation = train.train_test_split()

        self._scale_input_features()
        self._update_loaders()

    def _inverse_transform_target(self, y_scaled):
        """Convert normalized target (y_scaled) back to original units."""
        return y_scaled * (self.target_std + self.eps) + self.target_mean

    def _scale_target_feature(self):
        y_train = self._training._y
        self.target_mean = y_train.mean().item()
        self.target_std = y_train.std().item()

        self._training._y = (self._training._y - self.target_mean) / (self.target_std + self.eps)
        self._validation._y = (self._validation._y - self.target_mean) / (self.target_std + self.eps)
        self._test._y = (self._test._y - self.target_mean) / (self.target_std + self.eps)

    def _scale_input_features(self):
        n_discrete_features = len(set(self._features) & set(DISCRETE_FEATURES))
        self._scaler.fit(self._training.scalable_data,n_discrete_features)
        self._scale_target_feature()
        self._training.transform(self._scaler,n_discrete_features)
        self._validation.transform(self._scaler,n_discrete_features)
        self._test.transform(self._scaler,n_discrete_features)
    
    def _update_loaders(self):
        self._train_loader = DataLoader(self._training, batch_size=self._batch_size, shuffle=False)
        self._val_loader = DataLoader(self._validation, batch_size=self._batch_size, shuffle=False)
        self._test_loader = DataLoader(self._test, batch_size=self._batch_size, shuffle=False)

    def remove_feature(self, feature):
        self._training.remove_feature(feature)
        self._validation.remove_feature(feature)
        self._test.remove_feature(feature)
        self._features.remove(feature)
        self._update_loaders()

    def get_feature_at_index(self, idx):
        return self._features[idx]

    @property
    def features(self):
        return self._features

    @property
    def train_loader(self):
        return self._train_loader
    
    @property
    def validation_loader(self):
        return self._val_loader
    
    @property
    def test_loader(self):
        return self._test_loader

if __name__ == "__main__":
    dat_loader = DataLoaderTimeSeries(48)
    feature = dat_loader.get_feature_at_index(0)
    dat_loader.remove_feature(feature)  