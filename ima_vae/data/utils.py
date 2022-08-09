from os import makedirs
from os.path import dirname, abspath, join, isdir
from os.path import isfile
from typing import Literal

import numpy as np
import torch
from matplotlib import pyplot as plt

from spriteworld.gen_sprites_dataset import sprites_gen_wrapper
from spriteworld.utils import sprites_filename

from ima_vae.metrics.cima import cima_kl_diagonality


def l2_normalize(Amat, axis=0):
    l2norm = np.sqrt(np.sum(Amat * Amat, axis))
    Amat = Amat / l2norm
    return Amat


def get_lin_mix(obs_dim, unit_det=False, col_norm=False):

    assert unit_det != col_norm
    rank = -1
    while rank != obs_dim:
        mat = np.random.rand(obs_dim, obs_dim)
        if col_norm is True:
            norm_mat = l2_normalize(mat, axis=0)
        elif unit_det is True:
            norm_mat = mat / np.power(np.abs(np.linalg.det(mat)), 1.0 / obs_dim)
        else:
            raise ValueError(
                f"Either unit_det or col_norm should be True, but both are False!"
            )

        try:
            rank = np.linalg.matrix_rank(norm_mat, tol=1e-6).item()
        except:
            rank = -1

    return torch.from_numpy(norm_mat).to(torch.float32)


def rand_cos_sim(v, costheta):
    # Form the unit vector parallel to v:
    u = v / np.linalg.norm(v)

    # Pick a random vector:
    r = np.random.multivariate_normal(np.zeros_like(v), np.eye(len(v)))

    # Form a vector perpendicular to v:
    uperp = r - r.dot(u) * u

    # Make it a unit vector:
    uperp = uperp / np.linalg.norm(uperp)

    # w is the linear combination of u and uperp with coefficients costheta
    # and sin(theta) = sqrt(1 - costheta**2), respectively:
    w = costheta * u + np.sqrt(1 - costheta**2) * uperp

    return w


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


def build_moebius_transform_torch(alpha, A, a, b, epsilon=2, linear_map=None):
    device = "cuda" if torch.cuda.is_available() is True else "cpu"
    A = A.to(device)
    a = a.to(device)
    b = b.to(device)
    if linear_map is not None:
        linear_map = linear_map.to(device)

    def mixing_moebius_transform(x):
        if epsilon == 2:
            frac = ((x - a) ** 2).sum()
            frac = frac ** (-1)
        else:
            frac = 1.0

        moebius = b + frac * alpha * (A @ (x - a).T).T

        if linear_map is not None:
            moebius = (linear_map @ moebius.T).T
        return moebius

    B = torch.linalg.inv(A)

    def unmixing_moebius_transform(y):
        numer = 1 / alpha * (y - b)
        if epsilon == 2:
            denom = ((numer) ** 2).sum()
        else:
            denom = 1.0

        inv_moebius = a + 1.0 / denom * (B @ numer.T).T
        if linear_map is not None:
            inv_moebius = (inv_moebius.T @ linear_map.inverse()).T

        return inv_moebius

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
    n_obs, n_classes, projective, affine, deltah, deltas, deltav, angle, shape, seed
):
    sprites_dir = join(
        join(dirname(dirname(dirname(abspath(__file__)))), "spriteworld"),
        "sprites_data",
    )

    hsv_change = deltah != 0 or deltas != 0 or deltav != 0

    path = join(
        sprites_dir,
        filename := sprites_filename(
            n_obs, n_classes, projective, affine, hsv_change, seed=seed
        ),
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
            seed=seed,
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
