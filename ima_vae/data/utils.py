from os import makedirs
from os.path import dirname, abspath, join, isdir
from os.path import isfile
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt

from spriteworld.gen_sprites_dataset import sprites_gen_wrapper
from spriteworld.utils import sprites_filename


def to_one_hot(x, m=None):
    "batch one hot"
    if type(x) is not list:
        x = [x]
    if m is None:
        ml = []
        for xi in x:
            ml += [xi.max() + 1]
        m = max(ml)
    dtp = x[0].dtype
    xoh = []
    for i, xi in enumerate(x):
        xoh += [np.zeros((xi.size, int(m)), dtype=dtp)]
        xoh[i][np.arange(xi.size), xi.astype(np.int)] = 1
    return xoh


def cart2pol(x, y):
    """
    From cartesian to polar coordinates
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def scatterplot_variables(x, title, colors="None", cmap="hsv"):
    if colors == "None":
        plt.scatter(x[:, 0], x[:, 1], color="r", s=30)
    else:
        plt.scatter(x[:, 0], x[:, 1], c=colors, s=30, alpha=0.75, cmap=cmap)

    plt.axis("off")
    plt.gca().set_aspect("equal", adjustable="box")


import torch


def build_moebius_transform_torch(alpha, A, a, b, epsilon=2):
    device = "cuda" if torch.cuda.is_available() is True else "cpu"
    A = A.to(device)
    a = a.to(device)
    b = b.to(device)

    def mixing_moebius_transform(x):
        if epsilon == 2:
            frac = ((x - a) ** 2).sum()
            frac = frac ** (-1)
        else:
            diff = (x - a).abs()

            frac = 1.0
        return b + frac * alpha * (A @ (x - a).T).T

    B = torch.linalg.inv(A)

    def unmixing_moebius_transform(y):
        numer = 1 / alpha * (y - b)
        if epsilon == 2:
            denom = ((numer) ** 2).sum()
        else:
            denom = 1.0
        return a + 1.0 / denom * (B @ numer.T).T

    return mixing_moebius_transform, unmixing_moebius_transform


def build_moebius_transform(alpha, A, a, b, epsilon=2):
    """
    Implements Möbius transformations for D>=2, based on:
    https://en.wikipedia.org/wiki/Liouville%27s_theorem_(conformal_mappings)

    alpha: a scalar
    A: an orthogonal matrix
    a, b: vectors in R^D (dimension of the data)
    """
    from jax import numpy as jnp

    def mixing_moebius_transform(x):
        if epsilon == 2:
            frac = jnp.sum((x - a) ** 2)
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


DatasetType = Literal["synth", "image"]


def load_sprites(
    n_obs, n_classes, projective, affine, deltah, deltas, deltav, angle, shape
):
    sprites_dir = join(dirname(abspath(__file__)), "sprites_data")
    hsv_change = deltah != 0 or deltas != 0 or deltav != 0

    path = join(
        sprites_dir,
        filename := sprites_filename(n_obs, n_classes, projective, affine, hsv_change),
    )

    if not isdir(sprites_dir):
        makedirs(sprites_dir)

    if not isfile(path):
        print("no dSprites file, generating data...")
        obs, labels, sources = sprites_gen_wrapper(
            nobs=n_obs,
            nclasses=n_classes,
            projective=projective,
            affine=affine,
            deltah=deltah,
            deltas=deltas,
            deltav=deltav,
            angle=angle,
            shape=shape,
            lower=2,
            upper=15,
        )

    else:
        obs = np.load(path)["arr_0"]
        labels = np.load(path)["arr_1"]
        sources = np.load(path)["arr_2"]

    mixing, unmixing = None, None

    discrete_list = [False] * 4
    if "angle" in filename:
        discrete_list.append(True)
    if "shape" in filename:
        discrete_list.append(False)

    return labels, obs, sources, mixing, unmixing, discrete_list
