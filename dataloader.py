from torch.utils.data import DataLoader
import pandas as pd
from dataset import TimeSeriesDataset
from scaler import MinMaxScaler
from one_hot_encode import one_hot_encode

class DataLoaderTimeSeries:
    def __init__(self, input_window:{24,48,72}, output_window = 24, batch_size = 128):
        self._input_window = input_window
        self._output_window = output_window
        self._batch_size = batch_size
        self._scaler = MinMaxScaler()

        self._dataset = self._get_dataset()
        self._features = list(self._dataset.columns.values)

        df_one_hot_encoded = one_hot_encode("Holiday_ID", self._dataset["Holiday_ID"])
        self._dataset = pd.concat([self._dataset, df_one_hot_encoded], axis=1)

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

    def _scale_input_features(self):
        flattened_X = self._training.X.view(-1,len(self._features))
        self._scaler.fit(flattened_X)
        self._training.transform(self._scaler)
        self._validation.transform(self._scaler)
        self._test.transform(self._scaler)
    
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


