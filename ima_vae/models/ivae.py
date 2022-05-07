import torch
from torch import nn

from ima_vae.data.utils import DatasetType
from ima_vae.distributions import Normal, Uniform, Beta, Laplace
from ima_vae.models import nets
from ima_vae.models.utils import weights_init, PriorType


class iVAE(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        data_dim: int,
        n_segments: int,
        n_classes: int,
        n_layers: int,
        activation,
        device,
        prior: PriorType = "uniform",
        likelihood=None,
        posterior=None,
        slope: float = 0.2,
        diag_posterior: bool = True,
        dataset: DatasetType = "synth",
        fix_prior=True,
        beta: float = 1.0,
        prior_alpha: float = 3.0,
        prior_beta: float = 11.0,
        prior_mean: float = 0.0,
        prior_var: float = 1.0,
        decoder_var: float = 0.000001,
    ):
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

        self._setup_distributions(
            likelihood,
            posterior,
            prior,
            device,
            diag_posterior,
            prior_alpha,
            prior_beta,
            prior_mean,
            prior_var,
        )
        self._setup_nets(dataset, device, n_layers, slope, decoder_var)

        self.interp_sample = None
        self.interp_dir = None
        self.apply(weights_init)

    def _setup_nets(self, dataset, device, n_layers, slope, decoder_var=0.000001):
        # decoder params
        self.decoder_var = decoder_var * torch.ones(1, dtype=torch.float64).to(device)

        if dataset == "synth":
            self.encoder, self.decoder = nets.get_synth_models(
                self.data_dim,
                self.latent_dim,
                self.post_dim,
                n_layers,
                self.activation,
                device,
                slope,
            )
        elif dataset == "image":
            self.encoder, self.decoder = nets.get_sprites_models(
                self.latent_dim, self.post_dim, n_channels=3
            )

    def _setup_distributions(
        self,
        likelihood,
        posterior,
        prior: PriorType,
        device,
        diag_posterior,
        prior_alpha: float = 3.0,
        prior_beta: float = 11.0,
        prior_mean: float = 0.0,
        prior_var: float = 1.0,
    ):
        # prior_params
        self.prior_mean = prior_mean * torch.ones(1).to(device)
        self.prior_var = prior_var * torch.ones(1).to(device)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        if prior == "gaussian" or prior is None:
            self.prior = Normal(device=device, diag=True)
        elif prior == "beta":
            self.prior = Beta()
        elif prior == "uniform":
            self.prior = Uniform()
        elif prior == "laplace":
            self.prior = Laplace()
        else:
            self.prior = prior

        if self.prior.name != "uniform":
            self.conditioner = nets.MLP(
                self.n_classes,
                self.latent_dim * (2 - bool(self.fix_prior)),
                self.latent_dim * 4,
                self.n_layers,
                activation=self.activation,
                slope=self.slope,
                device=device,
            )

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
        mu = encoding[:, : self.latent_dim]
        log_var = cholesky_factors = encoding[:, self.latent_dim :]

        return mu, log_var, cholesky_factors

    def _encode(self, x):
        enc_mean, enc_logvar, enc_cholesky = self._encoder_params(x)

        if self.posterior.diag:
            latents = self.posterior.sample(enc_mean, enc_logvar.exp())
            log_qz_xu = self.posterior.log_pdf(latents, enc_mean, enc_logvar.exp())
        else:
            self.cholesky = torch.zeros(
                (x.shape[0], self.latent_dim, self.latent_dim)
            ).to(x.device)
            self._populate_cholesky(enc_cholesky)
            latents = self.posterior.sample(enc_mean, self.cholesky)
            log_qz_xu = self.posterior.log_pdf_full(latents, enc_mean, self.cholesky)

        if self.prior.name == "beta" or self.prior.name == "uniform":
            eps = 1e-8
            latents = torch.sigmoid(latents)
            determ = torch.log(1.0 / (latents * (1.0 - latents) + eps)).sum(1)
            log_qz_xu += determ

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
        (
            encoding_mean,
            encoding_logvar,
            latents,
            reconstructions,
            log_qz_xu,
        ) = self.forward(x)

        log_px_z = self._obs_log_likelihood(reconstructions, x)
        # log_px_z = ((x_recon-x.to(device))**2).flatten(1).sum(1).mul(-1)

        log_pz_u, mean, var = self._prior_log_likelihood(latents, u)

        if self.prior.name == "gauss":
            from torch.distributions import kl_divergence, MultivariateNormal

            kl_loss = -torch.stack(
                [
                    torch.distributions.kl_divergence(
                        MultivariateNormal(
                            q_mean,
                            torch.diag(q_var),
                        ),
                        MultivariateNormal(p_mean, torch.diag(p_var)),
                    )
                    for q_mean, q_var, p_mean, p_var in zip(
                        encoding_mean, encoding_logvar.exp(), mean, var
                    )
                ]
            ).mean()
        else:
            kl_loss = (log_pz_u - log_qz_xu).mean()
        rec_loss = log_px_z.mean()

        if log is True:
            latent_stat = self._latent_statistics(encoding_mean, encoding_logvar.exp())

        neg_elbo = -(rec_loss + self.beta * kl_loss)

        return (
            neg_elbo,
            latents,
            rec_loss,
            kl_loss,
            None if log is False else latent_stat,
            None if reconstruction is False else reconstructions,
        )

    def _prior_log_likelihood(self, latents, u):
        # all prior parameters fixed if uniform
        if self.prior.name == "uniform":
            mean, var = self.prior_mean, self.prior_var

        else:
            prior_params = self.conditioner(u)

            if self.fix_prior is False:
                prior_mean = prior_params[:, : self.latent_dim]
                prior_logvar = prior_params[:, self.latent_dim :]

            if self.prior.name == "gauss":
                if self.fix_prior is True:
                    mean, var = self.prior_mean, prior_params.exp()

                else:
                    mean, var = prior_mean, prior_logvar.exp()

            elif self.prior.name == "beta":
                if self.fix_prior is True:
                    mean = (
                        torch.ones(
                            (latents.shape[0], self.latent_dim), device=latents.device
                        )
                        * self.prior_alpha
                    )
                    var = (
                        torch.ones(
                            (latents.shape[0], self.latent_dim), device=latents.device
                        )
                        * self.prior_beta
                    )

                else:
                    mean = torch.abs(prior_mean) + 2
                    var = torch.abs(prior_logvar) + 2

            elif self.prior.name == "laplace":
                if self.fix_prior is True:
                    mean, var = self.prior_mean, self.prior_var
                else:
                    mean, var = prior_mean, prior_logvar.exp()

        log_pz_u = self.prior.log_pdf(latents, mean, var)
        return log_pz_u, mean, var

    def _obs_log_likelihood(self, reconstructions, x):
        log_px_z = self.likelihood.log_pdf(
            reconstructions.flatten(1), x.flatten(1), self.decoder_var
        )
        return log_px_z

    def _latent_statistics(self, encoding, enc_variance) -> dict:

        latent_variance = enc_variance.mean(0)
        latent_mean = encoding.mean(0)
        latent_stat = {
            **{
                f"latent_variance_{i}": latent_variance[i] for i in range(self.data_dim)
            },
            **{f"latent_mean_{i}": latent_mean[i] for i in range(self.data_dim)},
            **{f"latent_mean_variance": latent_mean.var()},
        }

        return latent_stat

    def _populate_cholesky(self, cholesky_factors):
        it = 0
        for i in range(self.cholesky.shape[1]):
            for j in range(i + 1):
                self.cholesky[:, i, j] = cholesky_factors[:, it]
                it += 1
