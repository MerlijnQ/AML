import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, dataset_file_name):
        self.dataset = pd.read_csv(dataset_file_name)

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        pass

test_dataset = MyDataset("dataset/continuous dataset.csv")
print(len(test_dataset))