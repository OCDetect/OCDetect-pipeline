import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

class OCDetectDataset(Dataset):
    preloaded_data = None

    @classmethod
    def preload(cls, windows, users, labels):
        cls.preloaded_data = dict()
        if type(users) == pd.DataFrame:
            users = users["user"]
        if type(labels) == pd.DataFrame:
            labels = labels["0"]

        for user in users.unique():
            cls.preloaded_data[user] = [[],[]]
        for i, window in enumerate(windows):
            user = users[i]
            cls.preloaded_data[user][0].append(window)
            cls.preloaded_data[user][1].append(labels[i])
        for user in users.unique():
            cls.preloaded_data[user][0] = np.stack(cls.preloaded_data[user][0])
            cls.preloaded_data[user][1] = np.array(cls.preloaded_data[user][1])

    def __init__(self, subjects, window_size=150, model=""):
        if OCDetectDataset.preloaded_data is None:
            raise Exception("DatasetClass must be initialized using preload first!")
        self.channels = 6
        self.window_size = window_size
        self.model_name = model
        self.classes = 3
        ### get preloaded data ###
        windows = []
        labels = []
        for subject in subjects:
            windows.append(OCDetectDataset.preloaded_data[subject][0])
            labels.append(OCDetectDataset.preloaded_data[subject][1])
        self.features = np.concatenate(windows)
        self.labels = np.concatenate(labels)
        self.length = len(self.features)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        reduce = False
        if type(idx) == int:
            idx = slice(idx, idx + 1)
            reduce = True
        if reduce:
            return torch.FloatTensor(self.features[idx])[0], torch.LongTensor(self.labels[idx])[0]
        return torch.FloatTensor(self.features[idx]), torch.LongTensor(self.labels[idx])
