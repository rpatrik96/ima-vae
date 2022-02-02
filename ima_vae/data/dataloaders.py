import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BasicDataset(Dataset):
    def __init__(self, lname=None, obs=None, labels=None, sources=None):
        if lname:
            self.obs = torch.from_numpy(np.load(lname)['arr_0'])
            self.labels = torch.from_numpy(np.load(lname)['arr_1'])
            self.sources = torch.from_numpy(np.load(lname)['arr_2'])
        else:
            self.obs = torch.from_numpy(obs)
            self.labels = torch.from_numpy(labels)
            self.sources = torch.from_numpy(sources)

        self.len = self.obs.shape[0]
        self.aux_dim = self.y.shape[1]
        self.data_dim = self.obs.shape[1]
        self.latent_dim = self.data_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.obs[index], self.labels[index], self.sources[index]

    def get_dims(self):
        return self.data_dim, self.latent_dim, self.aux_dim


def get_dataloader(batch_size, obs=None, labels=None, sources=None, lname=None):
    if lname:
        return DataLoader(BasicDataset(lname=lname), batch_size=batch_size, shuffle=True)
    else:
        return DataLoader(BasicDataset(obs=obs, labels=labels, sources=sources), batch_size=batch_size,
                          shuffle=True)
