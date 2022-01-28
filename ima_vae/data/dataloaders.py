import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class BasicDataset(Dataset):
    def __init__(self, lname=None, X=None, Y=None, S=None, transform=None):
        if lname:
            self.x = np.load(lname)['arr_0']
            self.y = np.load(lname)['arr_1']
            self.s = np.load(lname)['arr_2']
        else:
            self.x = X
            self.y = Y
            self.s = S
        if transform:
            self.transform = transforms.ToTensor()
        else:
            self.transform = None

        self.len = self.x.shape[0]
        #self.aux_dim = self.y.shape[1]
        self.data_dim = self.x.shape[1]
        self.latent_dim = self.data_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = self.x[index]  
        if self.transform:
            x = self.transform(x)
        return x, self.y[index], self.s[index]

    def get_dims(self):
        return self.data_dim, self.latent_dim#, self.aux_dim


def get_dataloader(batch_size,X=None,Y=None,S=None,lname=None,transform=False):
    if lname:
        return DataLoader(BasicDataset(lname=lname, transform=transform),batch_size=batch_size,shuffle=True)
    else:
        return DataLoader(BasicDataset(X=X,Y=Y,S=S,transform=transform),batch_size=batch_size,shuffle=True)
    