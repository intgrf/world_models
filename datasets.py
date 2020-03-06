from torch.utils.data import Dataset
from os.path import join, isdir
from os import listdir
import numpy as np
from bisect import bisect


class RolloutDataset(Dataset):
    def __init__(self, path, transform, seq_len, buffer_size, train=True):
        super(RolloutDataset, self).__init__()
        self.transform = transform

        self.files = [join(path, sd) for sd in listdir(path)]

        if train:
            self.files = self.files[:-600]
        else:
            self.files = self.files[-600:]

        self.buffer = None
        self.buffer_files = None
        self.cum_size = None
        self.buffer_index = 0
        self.buffer_size = buffer_size
        self.seq_len = seq_len

    def load_next_buffer(self):
        self.buffer_files = self.files[self.buffer_index : self.buffer_index + self.buffer_size]
        self.buffer_index += self.buffer_size
        self.buffer_index = self.buffer_index % len(self.files)
        self.buffer = []
        self.cum_size = [0]

        for file in self.buffer_files:
            with np.load(file) as data:
               self.buffer += [{k: np.copy(v) for k, v in data.items()}]
               self.cum_size += [self.cum_size[-1] +
                                  data['rewards'].shape[0]]

    def __len__(self):
        if not self.cum_size:
            self.load_next_buffer()
        return self.cum_size[-1]

    def __getitem__(self, item):
        file_idx = bisect(self.cum_size, item) - 1
        seq_idx = item - self.cum_size[file_idx]
        data = self.buffer[file_idx]
        return self.get_data(data, seq_idx)

    def get_data(self, data, seq_idx):
        states_data = data['states'][seq_idx : seq_idx + self.seq_len + 1]
        states_data = self.transform(states_data.astype(np.float32))
        state, next_state = states_data[:-1], states_data[1:]
        action = data['actions'][seq_idx + 1 : seq_idx + self.seq_len + 1].astype(np.float32)
        reward = data['rewards'][seq_idx + 1 : seq_idx + self.seq_len + 1].astype(np.float32)
        done = data['dones'][seq_idx + 1 : seq_idx + self.seq_len + 1].astype(np.float32)

        return state, action, reward, next_state, done


class ObservationDataset(Dataset):
    def __init__(self, path, transform, buffer_size=200, train=True):
        super(ObservationDataset, self).__init__()
        self.transform = transform

        self.files = [join(path, sd) for sd in listdir(path)]

        if train:
            self.files = self.files[:-150]
        else:
            self.files = self.files[-150:]

        self.buffer = None
        self.buffer_files = None
        self.cum_size = None
        self.buffer_index = 0
        self.buffer_size = buffer_size

    def load_next_buffer(self):
        self.buffer_files = self.files[self.buffer_index : self.buffer_index + self.buffer_size]
        self.buffer_index += self.buffer_size
        self.buffer_index = self.buffer_index % len(self.files)
        self.buffer = []
        self.cum_size = [0]

        for file in self.buffer_files:
            with np.load(file) as data:
               self.buffer += [{k: np.copy(v) for k, v in data.items()}]
               self.cum_size += [self.cum_size[-1] +
                                  data['rewards'].shape[0]]

    def __len__(self):
        if not self.cum_size:
            self.load_next_buffer()
        return self.cum_size[-1]

    def __getitem__(self, item):
        file_idx = bisect(self.cum_size, item) - 1
        seq_idx = item - self.cum_size[file_idx]
        data = self.buffer[file_idx]
        return self.get_data(data, seq_idx)

    def get_data(self, data, seq_idx):
        return self.transform(data['states'][seq_idx])