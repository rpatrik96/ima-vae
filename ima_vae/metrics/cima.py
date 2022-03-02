import torch
from jax import numpy as jnp


def cima_kl_diagonality(jacobian: torch.Tensor):
    """
    Calculates the IMA constrast. Able to handle jax and Pytorch objects as well

    :param jacobian: jacobian matrix (jax or Pytorch)
    :return:
    """
    jacobian_t_jacobian = jacobian.T @ jacobian

    lib = torch if type(jacobian) is torch.Tensor else jnp

    return 0.5 * (
        lib.linalg.slogdet(lib.diag(lib.diag(jacobian_t_jacobian)))[1]
        - lib.linalg.slogdet(jacobian_t_jacobian)[1]
    )
