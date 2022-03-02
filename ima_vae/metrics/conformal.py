import torch


def frobenius_diagonality(matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Frobenius measure of diagonality for correlation matrices.
    Source: https://www.sciencedirect.com/science/article/pii/S0024379516303834#se0180

    :param matrix: matrix as a torch.Tensor
    :return:
    """

    return 0.5 * (
        (matrix - torch.eye(matrix.shape[0], device=matrix.device)).norm("fro").pow(2)
    )


def conformal_contrast(jacobian: torch.Tensor) -> torch.Tensor:
    JJ_T = jacobian @ jacobian.T

    max_abs_scale = JJ_T.abs().diag().max()

    return frobenius_diagonality(JJ_T / max_abs_scale)


def col_norm_var(jacobian: torch.Tensor) -> torch.Tensor:
    col_norms = jacobian.norm(p=2, dim=0)

    return col_norms.var()
