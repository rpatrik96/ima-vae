import numpy as np
import torch
from torch import distributions as dist
from torch import nn

from ima_vae.models.nets import MLP
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)




from ima_vae.distributions import Normal


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

    def elbo(self, x, u, log=True, reconstruction: bool = False):
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

        return rec_loss + kl_loss, latent, rec_loss, kl_loss, None if log is False else latent_stat, None if reconstruction is False else \
        decoder_params[0]

    def _latent_statistics(self, encoding, enc_variance) -> dict:

        latent_mean_variance = enc_variance.mean(0)
        latent_mean = encoding.mean(0)
        latent_stat = {**{f"latent_mean_variance_{i}": latent_mean_variance[i] for i in range(self.data_dim)},
                       **{f"latent_mean_{i}": latent_mean[i] for i in range(self.data_dim)}}

        return latent_stat
