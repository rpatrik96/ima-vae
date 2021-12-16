import jax.numpy as jnp
import numpy as np
from jax import jacfwd
from scipy.stats import ortho_group

from ima_vae.data.data_generators import build_moebius_transform


def test_moebius_orthogonality(args):
    # generate moebius transform
    alpha = 1.0
    mixing_matrix = ortho_group.rvs(args.latent_dim)
    a = np.random.rand(args.latent_dim)
    b = np.zeros(args.latent_dim)
    mixing_moebius, _ = build_moebius_transform(alpha, mixing_matrix, a, b)

    # calculate jacobian
    jacobian = jacfwd(mixing_moebius)(a / 2)

    q, _ = jnp.linalg.qr(jacobian)

    # as the jacobian should be column-orthogonal,
    # the Q matrix of the QR-decomposition should have a |det| = 1
    assert abs(abs(jnp.linalg.det(q)) - 1) < 1e-5
