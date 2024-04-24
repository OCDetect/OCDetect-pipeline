import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split as tts


class OCDetectDataset(Dataset):
    preloaded_data = None

    @classmethod
    def preload(cls, windows, users, labels):
        cls.preloaded_data = dict()
        if type(users) == pd.DataFrame:
            users = users["user"]
        if type(labels) == pd.DataFrame:
            try:
                labels = labels["0"]
            except:
                labels = labels[0]

        for user in users.unique():
            cls.preloaded_data[user] = [[],[]]
        for i, window in enumerate(windows):
            user = users[i]
            cls.preloaded_data[user][0].append(window)
            cls.preloaded_data[user][1].append(labels[i])
        for user in users.unique():
            cls.preloaded_data[user][0] = np.stack(cls.preloaded_data[user][0])
            cls.preloaded_data[user][1] = np.array(cls.preloaded_data[user][1])

    def __init__(self, subjects, window_size=250, model="", retrain=False, idx=None):
        if OCDetectDataset.preloaded_data is None:
            raise Exception("DatasetClass must be initialized using preload first!")
        self.channels = 6
        self.window_size = window_size
        self.model_name = model
        self.classes = 2
        ### get preloaded data ###
        windows = []
        labels = []
        for subject in subjects:
            windows.append(OCDetectDataset.preloaded_data[subject][0])
            labels.append(OCDetectDataset.preloaded_data[subject][1])

        self.features = np.concatenate(windows)
        self.labels = np.concatenate(labels)
        if retrain:  # Select n_samples occurrences of 1 and same amount of 0 labels
            assert len(subjects) == 1
            n_samples = 50
            if idx is None:
                feat_train, _, label_train, _, idx, _ = tts(self.features, self.labels,
                                                                np.arange(len(self.features)), stratify=self.labels,
                                                            train_size=0.3)

                self.idx = idx
            else:
                # use the inverse of the already used idx:
                tmp_idx = np.ones(len(self.features), np.bool)
                tmp_idx[idx] = 0
                idx = tmp_idx
            self.features = self.features[idx]
            self.labels = self.labels[idx]
        self.length = len(self.features)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if type(idx) == int:
            idx = slice(idx, idx + 1)
            return torch.FloatTensor(self.features[idx])[0], torch.LongTensor(self.labels[idx])[0]
        return torch.FloatTensor(self.features[idx]), torch.LongTensor(self.labels[idx])