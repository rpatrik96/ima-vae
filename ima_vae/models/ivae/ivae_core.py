from typing import Literal

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

ActivationType = Literal['lrelu', 'sigmoid', 'none']


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation: ActivationType, device, slope=.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        self.activation = [activation] * (self.n_layers - 1)

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h


class Dist:
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass


class Normal(Dist):
    def __init__(self, device='cpu', diag=True):
        super().__init__()
        self.device = device
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        self._dist = dist.normal.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        self.name = 'gauss'
        self.diag = True

    def sample(self, mu, v):
        eps = self._dist.sample(mu.size()).squeeze()
        std = v.sqrt()
        scaled = eps.mul(std)
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


class iVAE(nn.Module):
    def __init__(self, latent_dim, data_dim, n_segments, n_layers, hidden_dim, activation, device, prior=None,
                 decoder=None, encoder=None, slope=.2):
        super().__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.n_segments = n_segments
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.cholesky_factors = int((latent_dim * (latent_dim + 1)) / 2)
        self.activation = activation
        self.slope = slope

        self.setup_distributions(decoder, device, encoder, prior)

        # prior_params
        self.prior_mean = torch.zeros(1).to(device)
        self.log_likelihood = MLP(n_segments, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                                  device=device)

        # decoder params
        self.decoder = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                           device=device)
        self.decoder_var = .01 * torch.ones(1).to(device)

        # encoder params
        self.encoder = MLP(data_dim + n_segments, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                           device=device)
        self.log_var = MLP(data_dim + n_segments, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                           device=device)

        self.apply(weights_init)

    def setup_distributions(self, decoder, device, encoder, prior):
        if prior is None:
            self.prior_dist = Normal(device=device, diag=True)
        else:
            self.prior_dist = prior
        if decoder is None:
            self.decoder_dist = Normal(device=device, diag=True)
        else:
            self.decoder_dist = decoder
        if encoder is None:
            self.encoder_dist = Normal(device=device, diag=True)
        else:
            self.encoder_dist = encoder

    def encoder_params(self, x, u):
        """

        :param x: observations
        :param u: segment labels
        :return:
        """
        xu = torch.cat((x, u), 1)
        encoding = self.encoder(xu)
        log_var = self.log_var(xu)
        return encoding, log_var.exp()

    def decoder_params(self, latent):
        decoding = self.decoder(latent)
        return decoding, self.decoder_var

    def prior_params(self, u):
        """

        :param u: segment labels
        :return:
        """
        logl = self.log_likelihood(u)
        return self.prior_mean, logl.exp()

    def forward(self, x, u):
        """

        :param x: observations
        :param u: segment labels
        :return:
        """
        prior_params = self.prior_params(u)
        encoder_params = self.encoder_params(x, u)
        latent = self.encoder_dist.sample(*encoder_params)
        decoder_params = self.decoder_params(latent)
        return decoder_params, encoder_params, latent, prior_params

    def elbo(self, x, u, log=True):
        """

        :param x: observations
        :param u: segment labels
        :return:
        """
        decoder_params, (encoding, enc_variance), latent, prior_params = self.forward(x, u)
        log_px_z = self.decoder_dist.log_pdf(x, *decoder_params)
        log_qz_xu = self.encoder_dist.log_pdf(latent, encoding, enc_variance)
        log_pz_u = self.prior_dist.log_pdf(latent, *prior_params)

        kl_loss = (log_pz_u - log_qz_xu).mean()
        rec_loss = log_px_z.mean()

        if log is True:
            latent_stat = self._latent_statistics(encoding, enc_variance)

        return rec_loss + kl_loss, latent, rec_loss, kl_loss, None if log is False else latent_stat

    def _latent_statistics(self, encoding, enc_variance) -> dict:

        latent_mean_variance = enc_variance.mean(0)
        latent_mean = encoding.mean(0)
        latent_stat = {**{f"latent_mean_variance_{i}": latent_mean_variance[i] for i in range(self.data_dim)},
                       **{f"latent_mean_{i}": latent_mean[i] for i in range(self.data_dim)}}

        return latent_stat
