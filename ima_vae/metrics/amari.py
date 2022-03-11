import torch


def amari_distance(W: torch.Tensor, A: torch.Tensor) -> float:
    """
    Computes the Amari distance between the products of two collections of matrices W and A.
    It cancels when the average of the absolute value of WA is a permutation and scale matrix.

    Based on the implementation of Amari distance in:
    https://github.com/pierreablin/picard/blob/master/picard/_tools.py

    Parameters
    ----------
    W : torch.Tensor, shape (n_features, n_features)
        Input collection of matrices
    A : torch.Tensor, shape (n_features, n_features)
        Input collection of matrices
    Returns
    -------
    d : torch.Tensor, shape (1, )
        The average Amari distance between the average of absolute values of the products of W and A.
    """

    P = W @ A

    def s(r):
        return ((r**2).sum(axis=-1) / (r**2).max(axis=-1)[0] - 1).sum(axis=-1)

    return ((s(P.abs()) + s(P.permute(0, 2, 1).abs())) / (2 * P.shape[1])).mean().item()
