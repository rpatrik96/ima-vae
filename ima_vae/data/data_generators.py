import numpy as np
import torch
from scipy.stats import ortho_group
from sklearn.preprocessing import scale
from torch.utils.data import Dataset

from ima.ima.mixing_functions import build_moebius_transform
from .utils import to_one_hot


class ConditionalDataset(Dataset):
    """
    a Dataset object holding a tuple (x,y): observed and auxiliary variable
    used in `models.ivae.ivae_wrapper.IVAE_wrapper()`
    """

    def __init__(self, X, Y, S, device='cpu'):
        self.device = device
        self.obs = torch.from_numpy(X)
        self.labels = torch.from_numpy(Y)
        self.sources = torch.from_numpy(S)
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


def gen_data(Ncomp, Nlayer, Nsegment, NsegmentObs, orthog, seed, NonLin, source='Gaussian', negSlope=.2,
             Niter4condThresh=1e4, one_hot_labels=True, mobius=False):
    if NonLin == 'none':
        nlayers = 1
    else:
        nlayers = Nlayer

    np.random.seed(seed)

    # generate non-stationary data:
    Nobs = NsegmentObs * Nsegment  # total number of observations
    labels = np.array([0] * Nobs)  # labels for each observation (populate below)

    if source == 'Gaussian':
        sources = np.random.normal(0, 1, (Nobs, Ncomp))
    elif source == 'Laplace':
        sources = np.random.laplace(0, 1, (Nobs, Ncomp))
    sources = scale(sources)

    # get modulation parameters
    modMat = np.random.uniform(0.01, 3, (Ncomp, Nsegment))

    # now we adjust the variance within each segment in a non-stationary manner
    for seg in range(Nsegment):
        segID = range(NsegmentObs * seg, NsegmentObs * (seg + 1))
        sources[segID, :] = np.multiply(sources[segID, :], modMat[:, seg])
        labels[segID] = seg

    observations = np.copy(sources)

    np.random.seed(seed)

    if mobius:
        moeb_params = np.load('moebius_transform_params.npy', allow_pickle=True).item()
        alpha = 1.0
        mixing_matrix = ortho_group.rvs(Ncomp)
        a = np.array(moeb_params['a'])
        b = np.zeros(2)
        mixing_moebius = build_moebius_transform(alpha, mixing_matrix, a, b)
        observations = mixing_moebius(observations)
    else:
        for l in range(nlayers):
            if orthog:
                mixing_matrix = ortho_group.rvs(Ncomp)
            else:
                mixing_matrix = generateUniformMat(Ncomp)

            # Apply non-linearity
            if NonLin == 'lrelu':
                observations = leaky_ReLU(observations, negSlope)
            elif NonLin == 'sigmoid':
                observations = sigmoidAct(observations)
            # Apply mixing:
            observations = np.dot(observations, mixing_matrix)

    if one_hot_labels:
        labels = to_one_hot(labels)[0]

    return observations, labels, sources
