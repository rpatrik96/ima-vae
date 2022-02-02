from os.path import dirname, abspath, join

import numpy as np
import torch
from scipy.stats import ortho_group
from sklearn.preprocessing import scale
from torch.utils.data import Dataset

from .utils import to_one_hot


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


def leaky_ReLU_1d(d, negSlope):
    """
    one dimensional implementation of leaky ReLU
    """
    if d > 0:
        return d
    else:
        return d * negSlope


leaky1d = np.vectorize(leaky_ReLU_1d)


def sigmoidAct(x):
    """
    one dimensional application of sigmoid activation function
    """
    return 1. / (1 + np.exp(-1 * x))


def leaky_ReLU(D, negSlope):
    """
    implementation of leaky ReLU activation function
    """
    assert negSlope > 0  # must be positive
    return leaky1d(D, negSlope)


def generateUniformMat(Ncomp):
    """
    generate a random matrix by sampling each element uniformly at random
    check condition number versus a condition threshold
    """
    A = np.random.uniform(1, 3, (Ncomp, Ncomp)) - 1
    for i in range(Ncomp):
        A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    return A



import jax.numpy as jnp


# taken from IMA repo
def build_moebius_transform(alpha, A, a, b, epsilon=2):
    '''
    Implements Möbius transformations for D>=2, based on:
    https://en.wikipedia.org/wiki/Liouville%27s_theorem_(conformal_mappings)

    alpha: a scalar
    A: an orthogonal matrix
    a, b: vectors in \RR^D (dimension of the data)
    '''

    def mixing_moebius_transform(x):
        if epsilon == 2:
            frac = jnp.sum((x - a) ** 2)  # is this correct?
            frac = frac ** (-1)
        else:
            diff = jnp.abs(x - a)

            frac = 1.0
        return b + frac * alpha * (A @ (x - a).T).T

    B = jnp.linalg.inv(A)

    def unmixing_moebius_transform(y):
        numer = 1 / alpha * (y - b)
        if epsilon == 2:
            denom = jnp.sum((numer) ** 2)
        else:
            denom = 1.0
        return a + 1.0 / denom * (B @ numer.T).T

    return mixing_moebius_transform, unmixing_moebius_transform
