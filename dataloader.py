from torch.utils.data import DataLoader
import pandas as pd
from dataset import TimeSeriesDataset

class DataLoaderTimeSeries:
    def __init__(self, input_window:{24,48,72}, output_window = 24, batch_size = 128):
        self._input_window = input_window
        self._output_window = output_window
        self._batch_size = batch_size

        self._dataset = self._get_dataset()
        self._features = list(self._dataset.columns.values)
        self.remove_feature("datetime")

        self._update_loaders()

    def _get_dataset(self):
        path_dataset = "dataset/continuous dataset.csv"
        dataset = pd.read_csv(path_dataset)

        # discard data from the month covid lockdown happened.
        dataset = dataset[dataset["datetime"] <= "2020-03-01 00:00:00"]
        dataset["datetime"] = pd.to_datetime(dataset["datetime"])
        return dataset
    
    def _update_loaders(self):
        dat_time_series = TimeSeriesDataset(self._dataset, self._input_window, self._output_window, self._features)
        train, test = dat_time_series.train_test_split()
        training, validation = train.train_test_split()

        self._train_loader = DataLoader(training, batch_size=self._batch_size, shuffle=False)
        self._val_loader = DataLoader(validation, batch_size=self._batch_size, shuffle=False)
        self._test_loader = DataLoader(test, batch_size=self._batch_size, shuffle=False)
    

    def remove_feature(self, feature):
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
    
def get_normalized_dataset(dataset : TimeSeriesDataset, min_value_dataset = None, max_value_dataset = None):
    # Normalization using min-max normalization. NOTE: might want to use a different method!
    if min_value_dataset is None:
        min_value_dataset = dataset._orig_dataset['nat_demand'].min()
    if max_value_dataset is None:
        max_value_dataset = dataset._orig_dataset['nat_demand'].max()
    normalized_dataset = (dataset._orig_dataset['nat_demand'] - min_value_dataset) / (max_value_dataset - min_value_dataset)
    return normalized_dataset, min_value_dataset, max_value_dataset



if __name__ == "__main__":
    dat_loader = DataLoaderTimeSeries(48)
    feature = dat_loader.get_feature_at_index(0)
    dat_loader.remove_feature(feature)


