import numpy as np
import torch
from torch import distributions as dist


class Dist:
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass


class Beta(Dist):
    def __init__(self):
        super().__init__()
        self.name = 'beta'

    def log_pdf(self, z, alpha, beta):
        alpha, beta = torch.abs(alpha), torch.abs(beta)
        alpha, beta = torch.flatten(alpha), torch.flatten(beta)
        shape, zdim = z.shape[0], z.shape[1]
        concentration = torch.stack([alpha, beta], -1)
        z = torch.flatten(z)
        heads_tails = torch.stack([z, 1.0 - z], -1)
        log_p_z = (torch.log(heads_tails) * (concentration - 1.0)).sum(-1) + torch.lgamma(
            concentration.sum(-1)) - torch.lgamma(concentration).sum(-1)
        log_p_z = log_p_z.view(shape, zdim).sum(1)
        return log_p_z


class Normal(Dist):
    '''
    Code for Normal class adapted from: https://github.com/ilkhem/icebeem/blob/master/models/ivae/ivae_core.py
    '''

    def __init__(self, device='cpu', diag=True):
        super().__init__()
        self.device = device
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        self._dist = dist.normal.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        self.name = 'gauss'
        self.diag = True

    def sample(self, mu, v, diag=True):
        eps = self._dist.sample(mu.size()).squeeze()
        std = v.sqrt()
        if self.diag:
            scaled = eps.mul(std)
        else:
            # v is cholesky and not variance
            scaled = torch.matmul(v, eps.unsqueeze(2)).view(eps.shape)
        return scaled.add(mu)

    def log_pdf(self, x, mu, v):
        """compute the log-pdf of a normal distribution with diagonal covariance"""
        lpdf = -0.5 * (torch.log(self.c) + v.log() + (x - mu).pow(2).div(v))
        return lpdf.sum(dim=-1)

    def log_pdf_full(self, x, mu, v):
        """
        compute the log-pdf of a normal distribution with full covariance
        v is a batch of "pseudo sqrt" of covariance matrices of shape (batch_size, d_latent, d_latent)
        mu is batch of means of shape (batch_size, d_latent)
        """
        batch_size, d = mu.size()
        cov = torch.einsum('bik,bjk->bij', v, v)  # compute batch cov from its "pseudo sqrt"
        assert cov.size() == (batch_size, d, d)
        inv_cov = torch.inverse(cov)
        c = d * torch.log(self.c)
        _, logabsdets = torch.slogdet(cov)
        xmu = x - mu
        lpdf = -0.5 * (c + logabsdets + torch.einsum('bi,bij,bj->b', [xmu, inv_cov, xmu]))
        return lpdf


class Laplace(Dist):
    '''
    Code from: https://github.com/ilkhem/icebeem/blob/master/models/ivae/ivae_core.py
    '''

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self._dist = dist.laplace.Laplace(torch.zeros(1).to(self.device), torch.ones(1).to(self.device) / np.sqrt(2))
        self.name = 'laplace'

    def sample(self, mu, b):
        eps = self._dist.sample(mu.size())
        scaled = eps.mul(b)
        return scaled.add(mu)

    def log_pdf(self, x, mu, b, reduce=True, param_shape=None):
        """compute the log-pdf of a laplace distribution with diagonal covariance"""
        if param_shape is not None:
            mu, b = mu.view(param_shape), b.view(param_shape)
        lpdf = -torch.log(2 * b) - (x - mu).abs().div(b)
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf


class Uniform(Dist):
    def __init__(self):
        super().__init__()
        self.name = 'uniform'

    def log_pdf(self, x, low, high):
        lb = low.le(x).type_as(low)
        ub = high.gt(x).type_as(low)
        lpdf = torch.log(lb.mul(ub)) - torch.log(high - low)
        return lpdf.sum(dim=-1)
