import torch
from torch.utils.data import Dataset


class ConditionalDataset(Dataset):
    """
    a Dataset object holding a tuple (x,y): observed and auxiliary variable
    used in `models.ivae.ivae_wrapper.IVAE_wrapper()`
    """

    def __init__(self, obs, labels, sources):
        self.obs = torch.from_numpy(obs)
        self.labels = torch.from_numpy(labels)
        self.sources = torch.from_numpy(sources)
        self.len = self.obs.shape[0]
        self.aux_dim = self.labels.shape[1]
        self.data_dim = self.obs.shape[1]
        self.latent_dim = self.data_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.obs[index], self.labels[index], self.sources[index]

    def get_dims(self):
        return self.data_dim, self.latent_dim, self.aux_dim
