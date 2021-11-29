import numpy as np
import torch
from scipy.stats import ortho_group
from scipy.stats import random_correlation
from sklearn.preprocessing import scale
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from .utils import to_one_hot


class ConditionalDataset(Dataset):
    """
    a Dataset object holding a tuple (x,y): observed and auxiliary variable
    used in `models.ivae.ivae_wrapper.IVAE_wrapper()`
    """

    def __init__(self, X, Y, S, device='cpu'):
        self.device = device
        self.x = torch.from_numpy(X)
        self.y = torch.from_numpy(Y)
        self.s = torch.from_numpy(S)
        self.len = self.x.shape[0]
        self.aux_dim = self.y.shape[1]
        self.data_dim = self.x.shape[1]
        self.latent_dim = self.data_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.s[index]

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

# taken from IMA repo
def build_moebius_transform(alpha, A, a, b, epsilon=2):
    '''
    Implements Möbius transformations for D>=2, based on:
    https://en.wikipedia.org/wiki/Liouville%27s_theorem_(conformal_mappings)
    
    alpha: a scalar
    A: an orthogonal matrix
    a, b: vectors in RR^D (dimension of the data)
    '''
    def mixing_moebius_transform(x):
        frac = np.sum((x-a)**2 , axis=1) #is this correct?
        test = A @ torch.from_numpy(x - a).permute(1,0).numpy()
        test = torch.from_numpy(test).permute(1,0).numpy()
        return b + (alpha * test)*frac[:,np.newaxis]
        
    return mixing_moebius_transform


def gen_data(Ncomp, Nlayer, Nsegment, NsegmentObs, orthog, seed, NonLin, source='Gaussian', negSlope=.2, Niter4condThresh=1e4, one_hot_labels=True, mobius=False):
    
    if NonLin == 'none':
        nlayers = 1
    else:
        nlayers = Nlayer

    np.random.seed(seed)

    # generate non-stationary data:
    Nobs = NsegmentObs * Nsegment  # total number of observations
    Y = np.array([0] * Nobs)  # labels for each observation (populate below)

    if source=='Gaussian':
        S = np.random.normal(0, 1, (Nobs, Ncomp))
    elif source=='Laplace':
        S = np.random.laplace(0, 1, (Nobs, Ncomp))
    S = scale(S)

    # get modulation parameters
    modMat = np.random.uniform(0.01, 3, (Ncomp, Nsegment))

    # now we adjust the variance within each segment in a non-stationary manner
    for seg in range(Nsegment):
        segID = range(NsegmentObs * seg, NsegmentObs * (seg + 1))
        S[segID, :] = np.multiply(S[segID, :], modMat[:, seg])
        Y[segID] = seg

    X = np.copy(S)

    np.random.seed(seed)

    if mobius:
        moeb_params = np.load('moebius_transform_params.npy', allow_pickle=True).item()
        alpha = 1.0
        A = ortho_group.rvs(Ncomp)
        a = np.array(moeb_params['a'])
        b = np.zeros(2)
        mixing_moebius = build_moebius_transform(alpha, A, a, b)
        X = mixing_moebius(X)
    else:
        for l in range(nlayers):
            if orthog:
                A = ortho_group.rvs(Ncomp)
            else:
                A = generateUniformMat(Ncomp)

            # Apply non-linearity
            if NonLin == 'lrelu':
                X = leaky_ReLU(X, negSlope)
            elif NonLin == 'sigmoid':
                X = sigmoidAct(X)
            # Apply mixing:
            X = np.dot(X, A)

    if one_hot_labels:
        Y = to_one_hot(Y)[0]

    return X, Y, S    

