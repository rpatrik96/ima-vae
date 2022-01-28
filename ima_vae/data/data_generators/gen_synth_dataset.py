import numpy as np
import torch
from scipy.stats import ortho_group
from scipy.stats import random_correlation
from sklearn.preprocessing import scale
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import jax
from jax import numpy as jnp
from scipy.stats import ortho_group
import numpy as np
from utils import to_one_hot, cart2pol, scatterplot_variables, build_moebius_transform

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
    A = np.random.uniform(1, 3, (Ncomp, Ncomp))
    print(A)
    return A

def gen_data(Ncomp, Nlayer, Nsegment, NsegmentObs, orthog, seed, NonLin, source='gaussian', negSlope=.2, Niter4condThresh=1e4, one_hot_labels=True, mobius=False):
    np.random.seed(2*seed)
    if NonLin == 'none':
        nlayers = 1
    else:
        nlayers = Nlayer

    # generate non-stationary data:
    Nobs = NsegmentObs * Nsegment  # total number of observations
    Y = np.array([0] * Nobs)  # labels for each observation (populate below)
    S = np.zeros((Nobs,Ncomp))
    if source=='uniform':
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        S = jax.random.uniform(subkey, shape=(Nobs, Ncomp), minval=0.0, maxval=1.0)
        S -= .5

    # if gaussian we adjust the variance within each segment in a non-stationary manner
    for seg in range(Nsegment):
        segID = range(NsegmentObs * seg, NsegmentObs * (seg + 1))
        print(segID)
        if source=='gaussian':
            mean=np.random.uniform(0, 0)
            var = np.random.uniform(0.01, 3)
            S[segID, :] = np.random.normal(mean, var, (NsegmentObs, Ncomp))
        elif source=='laplace':
            mean=np.random.uniform(0, 0)
            var = np.random.uniform(0.01, 3)
            S[segID, :] = np.random.laplace(mean, var, (NsegmentObs, Ncomp))
        elif source=='beta':
            alpha = np.random.uniform(3, 11)
            beta = np.random.uniform(3, 11)
            key = jax.random.PRNGKey(seed)
            key, subkey = jax.random.split(key)
            S[segID, :] = jax.random.beta(subkey, alpha, beta, (NsegmentObs, Ncomp))
            S -= .5
        Y[segID] = seg

    X = np.copy(S)

    if mobius:
        # Plot the sources
        if Ncomp == 2:
            _, colors = cart2pol(X[:, 0], X[:, 1])
            scatterplot_variables(X, 'Sources (train)', colors=colors)
            plt.title('Ground Truth', fontsize=19)
            plt.savefig("Sources_mobius",dpi=150,bbox_inches='tight')
            plt.close()
        # Generate a random orthogonal matrix
        A = ortho_group.rvs(dim=Ncomp)
        A = jnp.array(A)
        # Scalar
        alpha = 1.0
        # Two vectors with data dimensionality
        a = []
        while len(a) < Ncomp:
            s = np.random.randn()
            if np.abs(s) > 0.5:
                a = a + [s]
        a = jnp.array(a) # a vector in \RR^D
        b = jnp.zeros(Ncomp) # a vector in \RR^D
        epsilon = 2

        mixing, unmixing = build_moebius_transform(alpha, A, a, b, epsilon=epsilon)
        mixing_batched = jax.vmap(mixing)

        X = mixing_batched(X)
        mean = jnp.mean(X, axis=0)
        std = jnp.std(X, axis=0)
        X -= mean

        if Ncomp == 2:
            scatterplot_variables(X, 'Observations (train)', colors=colors)
            plt.title('Observations', fontsize=19)
            plt.savefig("Observations_mobius",dpi=150,bbox_inches='tight')
            plt.close()
        X = np.array(X)
        S = np.array(S)
   
    else:
        X = np.array(X)
        S = np.array(S)
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

