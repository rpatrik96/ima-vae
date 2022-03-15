import numpy as np
from jax import jacfwd
from scipy.stats import ortho_group

from ima_vae.data.utils import build_moebius_transform
from ima_vae.metrics.cima import cima_kl_diagonality


def test_moebius_orthogonality(args):
    # generate moebius transform
    alpha = 1.0
    mixing_matrix = ortho_group.rvs(args.model.latent_dim)
    a = np.random.rand(args.model.latent_dim)
    b = np.zeros(args.model.latent_dim)
    mixing_moebius, _ = build_moebius_transform(alpha, mixing_matrix, a, b)

    # calculate jacobian
    jacobian = jacfwd(mixing_moebius)(a / 2)

    # as the jacobian should be column-orthogonal,
    # meaning that the IMA contrast should be 0
    assert cima_kl_diagonality(jacobian) < 1e-5
