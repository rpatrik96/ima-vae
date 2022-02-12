import torch
from torch import nn

from ima_vae.data.utils import DatasetType
from ima_vae.distributions import Normal, Uniform, Beta
from ima_vae.models import nets
from ima_vae.models.utils import weights_init


class iVAE(nn.Module):
    def __init__(self, latent_dim, data_dim, n_segments, n_classes, n_layers, activation, device, prior=None,
                 likelihood=None, posterior=None, slope=.2, diag_posterior: bool = True, dataset: DatasetType = "synth",
                 fix_prior=False, beta=1.):
        super().__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.n_segments = n_segments
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.n_classes = n_classes
        self.fix_prior = fix_prior
        self.beta = beta

        self._setup_distributions(likelihood, posterior, prior, device, diag_posterior)
        self._setup_nets(dataset, device, n_layers, slope)

        self.interp_sample = None
        self.interp_dir = None
        self.apply(weights_init)

    def _setup_nets(self, dataset, device, n_layers, slope):
        # decoder params
        self.decoder_var = .000001 * torch.ones(1).to(device)

        if dataset == 'synth':
            self.encoder, self.decoder = nets.get_synth_models(self.data_dim, self.latent_dim, self.post_dim,
                                                               n_layers, self.activation, device, slope)
        elif dataset == 'image':
            self.encoder, self.decoder = nets.get_sprites_models(self.latent_dim, self.post_dim, n_channels=3)

    def _setup_distributions(self, likelihood, posterior, prior, device, diag_posterior):
        # prior_params
        self.prior_mean = torch.zeros(1).to(device)
        self.prior_var = torch.ones(1).to(device)

        if prior == 'gaussian' or prior is None:
            self.prior = Normal(device=device, diag=True)
        elif prior == 'beta':
            self.prior = Beta()
        elif prior == 'uniform':
            self.prior = Uniform()
        else:
            self.prior = prior

        if self.prior.name != 'uniform':
            self.conditioner = nets.MLP(self.n_classes, self.latent_dim * (2 - bool(self.fix_prior)),
                                        self.latent_dim * 4, self.n_layers,
                                        activation=self.activation, slope=self.slope,
                                        device=device)

        # likelihood
        if likelihood is None:
            self.likelihood = Normal(device=device, diag=True)
        else:
            self.likelihood = likelihood

        # posterior
        if posterior is None:
            self.posterior = Normal(device=device, diag=diag_posterior)
        else:
            self.posterior = posterior

        if self.posterior.diag:
            self.post_dim = 2 * self.latent_dim
        else:
            self.cholesky_factors = (self.latent_dim * (self.latent_dim + 1)) / 2
            self.post_dim = int(self.latent_dim + self.cholesky_factors)
            self.cholesky = None

    def _encoder_params(self, x):
        """

        :param x: observations
        :return:
        """
        encoding = self.encoder(x)
        mu = encoding[:, :self.latent_dim]
        log_var = cholesky_factors = encoding[:, self.latent_dim:]

        return mu, log_var, cholesky_factors

    def _encode(self, x):
        enc_mean, enc_logvar, enc_cholesky = self._encoder_params(x)

        if self.posterior.diag:
            latents = self.posterior.sample(enc_mean, enc_logvar.exp())
            log_qz_xu = self.posterior.log_pdf(latents, enc_mean, enc_logvar.exp())
        else:
            self.cholesky = torch.zeros((x.shape[0], self.latent_dim, self.latent_dim)).to(x.device)
            self._populate_cholesky(enc_cholesky)
            latents = self.posterior.sample(enc_mean, self.cholesky)
            log_qz_xu = self.posterior.log_pdf_full(latents, enc_mean, self.cholesky)

        if self.prior.name == 'beta' or self.prior.name == 'uniform':
            determ = torch.log(1. / (torch.sigmoid(latents) * (1. - torch.sigmoid(latents)))).sum(1)
            log_qz_xu += determ
            latents = torch.sigmoid(latents)

        return enc_logvar, enc_mean, latents, log_qz_xu

    def forward(self, x):
        """

        :param x: observations
        :return:
        """
        enc_logvar, enc_mean, latents, log_qz_xu = self._encode(x)

        reconstructions = self.decoder(latents)
        return enc_mean, enc_logvar, latents, reconstructions, log_qz_xu

    def neg_elbo(self, x, u, log=True, reconstruction: bool = False):
        """

        :param x: observations
        :param u: segment labels
        :return:
        """
        encoding_mean, encoding_logvar, latents, reconstructions, log_qz_xu = self.forward(x)

        log_px_z = self._obs_log_likelihood(reconstructions, x)
        # log_px_z = ((x_recon-x.to(device))**2).flatten(1).sum(1).mul(-1)

        log_pz_u = self._prior_log_likelihood(latents, u)

        kl_loss = (log_pz_u - log_qz_xu).mean()
        rec_loss = log_px_z.mean()

        if log is True:
            latent_stat = self._latent_statistics(encoding_mean, encoding_logvar.exp())

        neg_elbo = -(rec_loss + self.beta * kl_loss)

        return neg_elbo, latents, rec_loss, kl_loss, None if log is False else latent_stat, None if reconstruction is False else \
            reconstructions

    def _prior_log_likelihood(self, latents, u):
        # all prior parameters fixed if uniform
        if self.prior.name == 'uniform':
            log_pz_u = self.prior.log_pdf(latents, self.prior_mean, self.prior_var)
        else:
            prior_params = self.conditioner(u)

            if self.fix_prior is False:
                prior_mean = prior_params[:, :self.latent_dim]
                prior_logvar = prior_params[:, self.latent_dim:]

            if self.prior.name == 'gauss':
                if self.fix_prior is True:
                    log_pz_u = self.prior.log_pdf(latents, self.prior_mean, prior_params.exp())
                else:
                    log_pz_u = self.prior.log_pdf(latents, prior_mean, prior_logvar.exp())

            elif self.prior.name == 'beta':
                if self.fix_prior is True:
                    log_pz_u = self.prior.log_pdf(latents, torch.ones((latents.shape[0], self.latent_dim)) * 3,
                                                  torch.ones((latents.shape[0], self.latent_dim)) * 11)
                else:
                    log_pz_u = self.prior.log_pdf(latents, prior_mean, prior_logvar)
        return log_pz_u

    def _obs_log_likelihood(self, reconstructions, x):
        log_px_z = self.likelihood.log_pdf(reconstructions.flatten(1), x.flatten(1), self.decoder_var)
        return log_px_z

    def _latent_statistics(self, encoding, enc_variance) -> dict:

        latent_mean_variance = enc_variance.mean(0)
        latent_mean = encoding.mean(0)
        latent_stat = {**{f"latent_mean_variance_{i}": latent_mean_variance[i] for i in range(self.data_dim)},
                       **{f"latent_mean_{i}": latent_mean[i] for i in range(self.data_dim)}}

        return latent_stat

    def _populate_cholesky(self, cholesky_factors):
        it = 0
        for i in range(self.cholesky.shape[1]):
            for j in range(i + 1):
                self.cholesky[:, i, j] = cholesky_factors[:, it]
                it += 1
