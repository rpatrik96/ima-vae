from os.path import dirname, abspath, join

import jax
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from jax import numpy as jnp
from scipy.stats import ortho_group

from carefl.main import dict2namespace
from carefl.models.carefl import CAREFL
from ima_vae.data.utils import (
    to_one_hot,
    cart2pol,
    scatterplot_variables,
    build_moebius_transform,
    build_moebius_transform_torch,
    rand_cos_sim,
    get_lin_mix,
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


def smooth_leaky_relu(x, alpha=1.0):
    r"""Custom activation function
    Source:
    https://stats.stackexchange.com/questions/329776/approximating-leaky-relu-with-a-differentiable-function
    """
    return alpha * x + (1 - alpha) * np.logaddexp(x, 0)


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
    nonlin,
    seed=1,
    source="gaussian",
    break_orthog=0.0,
    neg_slope=0.2,
    one_hot_labels=True,
    mobius=False,
    alpha_shape=np.random.uniform(1, 10),
    beta_shape=np.random.uniform(1, 10),
    mean=np.random.uniform(0, 0),
    var=np.random.uniform(0.01, 3),
    ar_flow=False,
    mlp=False,
    unit_det=True,
    col_norm=False,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if nonlin == "none":
        nlayers = 1
    else:
        nlayers = num_layer

    labels, obs, sources = non_stationary_data(
        alpha_shape,
        beta_shape,
        mean,
        num_dim if ar_flow is False else num_dim,
        num_segment,
        num_segment_obs,
        seed,
        source,
        var,
    )

    linear_map = None
    if mobius is True:
        mixing, obs, sources, unmixing, linear_map = gen_mobius(
            num_dim, obs, sources, break_orthog, unit_det, col_norm
        )
    elif ar_flow is True:
        mixing, obs, sources, unmixing = gen_ar_flow(num_dim, sources)
    elif mlp is True:
        mixing, obs, sources, unmixing = gen_mlp(
            neg_slope, nlayers, nonlin, num_dim, obs, orthog, sources
        )
    else:
        raise NotImplementedError("No mixing selected!")

    if one_hot_labels:
        labels = to_one_hot(labels)[0]

    return (
        np.asarray(obs.astype(np.float32)),
        np.asarray(labels.astype(np.float32)),
        np.asarray(sources.astype(np.float32)),
        mixing,
        unmixing,
        [False] * num_dim,
        linear_map,
    )


def gen_ar_flow(num_dim, sources):
    assert num_dim % 2 == 0 and num_dim > 2
    # configuration
    config_file = "simulations.yaml"
    config_dir = join(
        dirname(dirname(dirname(dirname(abspath(__file__))))), "carefl/configs"
    )
    config_path = join(config_dir, config_file)
    with open(config_path, "r") as f:
        print("loading config file: {}".format(config_path))
        config_raw = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2namespace(config_raw)
    config.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    # flow setup
    model: CAREFL = CAREFL(config)
    model.dim = num_dim
    model.flow = model._get_flow_arch()[0]
    # make observations
    obs = model._backward_flow(np.asarray(sources))
    # mixing/unmixing
    mixing = lambda s: model.flow.backward(s)[0][-1]
    unmixing = lambda x: model.flow.forward(x)[0][-1]
    return mixing, obs, sources, unmixing


def non_stationary_data(
    alpha_shape,
    beta_shape,
    mean,
    num_dim,
    num_segment,
    num_segment_obs,
    seed,
    source,
    var,
):
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
    return labels, obs, sources


import torch.nn as nn
from torch.nn import Module

from torch import Tensor


class LeakySoftPlus(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{LeakySoftPlus}(x) = \text{negative\_slope}*x + (1-\text{negative\_slope}) * \log(1+\exp(x))


    Args:
        alpha: Controls the angle of the negative slope. Default: 2e-1


    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input


    Examples::

        >>> m = LeakySoftPlus(0.1)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ["alpha"]
    alpha: float

    def __init__(
        self,
        alpha: float = 2e-1,
    ) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, input: Tensor) -> Tensor:
        return self.alpha * input + (1.0 - self.alpha) * torch.logaddexp(
            input, torch.zeros_like(input)
        )


def gen_mlp(neg_slope, nlayers, nonlin, num_dim, obs, orthog, sources):
    mixing = None
    unmixing = None
    obs = np.array(obs)
    sources = np.array(sources)
    torch_mixing = []

    # Apply non-linearity
    if nonlin == "lrelu":
        act = lambda x: leaky_ReLU(x, neg_slope)
        torch_act = nn.LeakyReLU(negative_slope=neg_slope)
    elif nonlin == "sigmoid":
        act = lambda x: sigmoidAct(x)
        torch_act = nn.Sigmoid()
    elif nonlin == "smooth_lrelu":
        torch_act = LeakySoftPlus(neg_slope)
        act = lambda x: smooth_leaky_relu(x, neg_slope)
    elif nonlin == "none":
        act = lambda x: x
        torch_act = nn.Identity()
    else:
        raise ValueError

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for l in range(nlayers):
        mixing_matrix = (
            ortho_group.rvs(num_dim) if orthog is True else generateUniformMat(num_dim)
        )

        # create the components for the torch mixing
        layer = nn.Linear(num_dim, num_dim, bias=False)
        layer.weight = nn.Parameter(
            torch.from_numpy(mixing_matrix.astype(np.float32)).to(device),
            requires_grad=False,
        )
        torch_mixing.append(layer)
        torch_mixing.append(torch_act.to(device))

        # Apply non-linearity
        obs = act(obs)
        # Apply mixing:
        obs = np.dot(obs, mixing_matrix)

    mixing = nn.Sequential(*torch_mixing)
    return mixing, obs, sources, unmixing


def gen_mobius(num_dim, obs, sources, break_orthog, unit_det=True, col_norm=False):
    # Plot the sources
    if num_dim == 2:
        _, colors = cart2pol(obs[:, 0], obs[:, 1])
        scatterplot_variables(obs, "Sources (train)", colors=colors)
        plt.title("Ground Truth", fontsize=19)
        plt.savefig("Sources_mobius", dpi=150, bbox_inches="tight")
        plt.close()
    # Generate a random orthogonal matrix
    mixing_matrix = ortho_group.rvs(dim=num_dim)
    # Scalar
    alpha_shape = 1.0
    # Two vectors with data dimensionality
    a = []  # a vector in \RR^D
    while len(a) < num_dim:
        s = np.random.randn()
        if np.abs(s) > 0.5:
            a = a + [s]
    b = jnp.zeros(num_dim)  # a vector in \RR^D
    epsilon = 2
    mixing, unmixing = build_moebius_transform(
        alpha_shape, jnp.array(mixing_matrix), jnp.array(a), b, epsilon=epsilon
    )
    mixing_batched = jax.vmap(mixing)
    obs = mixing_batched(obs)

    linear_map = torch.eye(num_dim)
    if break_orthog != 0.0:
        if num_dim == 2:
            v1 = np.array([1, 0])
            v2 = rand_cos_sim(v1, break_orthog)
            linear_map = torch.Tensor(np.stack([v1, v2]))
        else:
            linear_map = get_lin_mix(num_dim, unit_det=unit_det, col_norm=col_norm)

        print(linear_map)
        obs = linear_map.numpy() @ obs.T
        obs = obs.T

    mean = jnp.mean(obs, axis=0)
    obs -= mean
    if num_dim == 2:
        scatterplot_variables(obs, "Observations (train)", colors=colors)
        plt.title("Observations", fontsize=19)
        plt.savefig("Observations_mobius", dpi=150, bbox_inches="tight")
        plt.close()

    obs = np.array(obs)
    sources = np.array(sources)
    print("Regenerating Moebius mixing with torch")
    mixing, unmixing = build_moebius_transform_torch(
        alpha_shape,
        torch.Tensor(mixing_matrix),
        torch.Tensor(a),
        torch.zeros(num_dim),
        epsilon=epsilon,
        linear_map=linear_map,
    )

    return mixing, obs, sources, unmixing, linear_map
