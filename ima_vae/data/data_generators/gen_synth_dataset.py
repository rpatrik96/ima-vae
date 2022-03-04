import matplotlib.pyplot as plt
import jax
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from scipy.stats import ortho_group
from ima_vae.data.utils import (
    to_one_hot,
    cart2pol,
    scatterplot_variables,
    build_moebius_transform,
)


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
    return 1.0 / (1 + np.exp(-1 * x))


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


def gen_data(
    num_dim,
    num_layer,
    num_segment,
    num_segment_obs,
    orthog,
    seed,
    nonlin,
    source="gaussian",
    neg_slope=0.2,
    Niter4condThresh=1e4,
    one_hot_labels=True,
    mobius=False,
    alpha_shape=np.random.uniform(1, 10),
    beta_shape=np.random.uniform(1, 10),
    mean=np.random.uniform(0, 0),
    var=np.random.uniform(0.01, 3),
):
    np.random.seed(2 * seed)
    if nonlin == "none":
        nlayers = 1
    else:
        nlayers = num_layer

    # generate non-stationary data:
    Nobs = num_segment_obs * num_segment  # total number of observations
    labels = np.array([0] * Nobs)  # labels for each observation (populate below)
    sources = np.zeros((Nobs, num_dim))
    if source == "uniform":
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        sources = jax.random.uniform(
            subkey, shape=(Nobs, num_dim), minval=0.0, maxval=1.0
        )
        sources -= 0.5

    # if gaussian we adjust the variance within each segment in a non-stationary manner
    for seg in range(num_segment):
        segID = range(num_segment_obs * seg, num_segment_obs * (seg + 1))
        print(segID)
        if source == "gaussian":
            sources[segID, :] = np.random.normal(mean, var, (num_segment_obs, num_dim))
        elif source == "laplace":
            sources[segID, :] = np.random.laplace(mean, var, (num_segment_obs, num_dim))
        elif source == "beta":
            key = jax.random.PRNGKey(seed)
            key, subkey = jax.random.split(key)
            sources[segID, :] = jax.random.beta(
                subkey, alpha_shape, beta_shape, (num_segment_obs, num_dim)
            )
            sources -= 0.5
        labels[segID] = seg

    obs = np.copy(sources)

    if mobius is True:
        # Plot the sources
        if num_dim == 2:
            _, colors = cart2pol(obs[:, 0], obs[:, 1])
            scatterplot_variables(obs, "Sources (train)", colors=colors)
            plt.title("Ground Truth", fontsize=19)
            plt.savefig("Sources_mobius", dpi=150, bbox_inches="tight")
            plt.close()
        # Generate a random orthogonal matrix
        mixing_matrix = ortho_group.rvs(dim=num_dim)
        mixing_matrix = jnp.array(mixing_matrix)
        # Scalar
        alpha_shape = 1.0
        # Two vectors with data dimensionality
        a = []
        while len(a) < num_dim:
            s = np.random.randn()
            if np.abs(s) > 0.5:
                a = a + [s]
        a = jnp.array(a)  # a vector in \RR^D
        b = jnp.zeros(num_dim)  # a vector in \RR^D
        epsilon = 2

        mixing, unmixing = build_moebius_transform(
            alpha_shape, mixing_matrix, a, b, epsilon=epsilon
        )
        mixing_batched = jax.vmap(mixing)

        obs = mixing_batched(obs)
        mean = jnp.mean(obs, axis=0)
        std = jnp.std(obs, axis=0)
        obs -= mean

        if num_dim == 2:
            scatterplot_variables(obs, "Observations (train)", colors=colors)
            plt.title("Observations", fontsize=19)
            plt.savefig("Observations_mobius", dpi=150, bbox_inches="tight")
            plt.close()
        obs = np.array(obs)
        sources = np.array(sources)

    else:
        mixing = None
        unmixing = None

        obs = np.array(obs)
        sources = np.array(sources)
        for l in range(nlayers):
            if orthog:
                mixing_matrix = ortho_group.rvs(num_dim)
            else:
                mixing_matrix = generateUniformMat(num_dim)

            # Apply non-linearity
            if nonlin == "lrelu":
                obs = leaky_ReLU(obs, neg_slope)
            elif nonlin == "sigmoid":
                obs = sigmoidAct(obs)
            # Apply mixing:
            obs = np.dot(obs, mixing_matrix)
    if one_hot_labels:
        labels = to_one_hot(labels)[0]

    return (
        np.asarray(obs.astype(np.float32)),
        np.asarray(labels.astype(np.float32)),
        np.asarray(sources.astype(np.float32)),
        mixing,
        unmixing,
        [False] * num_dim,
    )
