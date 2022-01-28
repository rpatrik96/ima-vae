import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

class Dist:
    def __init__(self):
        pass

    def log_pdf(self, *args, **kwargs):
        pass

class Beta(Dist):
    def __init__(self):
        super().__init__()
        self.name = 'beta'

    def log_pdf(self, z, alpha, beta):
        alpha, beta = torch.abs(alpha)+2, torch.abs(beta)+2
        alpha, beta = torch.flatten(alpha), torch.flatten(beta)
        shape, zdim = z.shape[0], z.shape[1]
        concentration = torch.stack([alpha,beta], -1)
        z = torch.flatten(z)
        heads_tails = torch.stack([z, 1.0 - z], -1)
        log_p_z = (torch.log(heads_tails) * (concentration - 1.0)).sum(-1) + torch.lgamma(concentration.sum(-1))- torch.lgamma(concentration).sum(-1)
        log_p_z = log_p_z.view(shape,zdim).sum(1)
        return log_p_z

'''
Code for Normal class adapted from: https://github.com/ilkhem/icebeem/blob/master/models/ivae/ivae_core.py
'''
class Normal(Dist):
    def __init__(self, device, diag=True):
        super().__init__()
        self.device = device
        self.diag = diag
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        self._dist = dist.normal.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        self.name = 'gauss'

    def sample(self, mu, v, diag=True):
        eps = self._dist.sample(mu.size()).squeeze()
        std = v.sqrt()
        if self.diag:
            scaled = eps.mul(std)
        else:
            # v is cholesky and not variance
            scaled = torch.matmul(v,eps.unsqueeze(2)).view(eps.shape)
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
        #rand_ind = np.random.randint(low=0, high=cov.shape[0]-1)
        #print(cov[rand_ind])
        _, logabsdets = torch.slogdet(cov)
        xmu = x - mu
        lpdf = -0.5 * (c + logabsdets + torch.einsum('bi,bij,bj->b', [xmu, inv_cov, xmu]))
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


