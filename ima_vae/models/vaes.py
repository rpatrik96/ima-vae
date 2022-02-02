import torch
from ima_vae.models import nets
from ima_vae.data.datamodules import DatasetType
from torch import nn


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


class iVAE(nn.Module):
    def __init__(self, prior, posterior, likelihood, latent_dim, n_classes, dataset:DatasetType, activation, device, n_layers,
                 slope=.2, nc=3):
        super(iVAE, self).__init__()
        self.latent_dim = latent_dim
        self.obs_dim = latent_dim
        self.prior = prior
        self.posterior = posterior
        self.likelihood = likelihood
        if posterior.diag:
            self.post_dim = 2 * latent_dim
        else:
            self.cholesky_factors = (latent_dim * (latent_dim + 1)) / 2
            self.post_dim = int(latent_dim + self.cholesky_factors)
            self.cholesky = None

        if dataset == 'synth':
            self.encoder = nets.MLP(self.obs_dim, self.post_dim, latent_dim * 10, n_layers, activation=activation, slope=slope,
                                    device=device)
            self.decoder = nets.MLP(latent_dim, self.obs_dim, latent_dim * 10, n_layers, activation=activation, slope=slope,
                                    device=device)

        elif dataset == 'image':
            self.encoder, self.decoder = nets.get_sprites_models(self.latent_dim, self.post_dim, nc=3)

        if self.prior.name != 'uniform':
            self.conditioner = nets.MLP(n_classes, latent_dim * 2, latent_dim * 4, n_layers, activation=activation, slope=slope,
                                        device=device)

        self.interp_sample = None
        self.interp_dir = None
        self.iter = 0
        self.apply(weights_init)

    def populate_cholesky(self, cholesky_factors):
        it = 0
        for i in range(self.cholesky.shape[1]):
            for j in range(i + 1):
                self.cholesky[:, i, j] = cholesky_factors[:, it]
                it += 1

    def forward(self, x):
        params = self.encoder(x)
        mu = params[:, :self.latent_dim]

        if self.posterior.diag:
            logvar = params[:, self.latent_dim:]
            z = self.posterior.sample(mu, logvar.exp())
            log_qz_u = self.posterior.log_pdf(z, mu, logvar.exp())
        else:
            self.cholesky = torch.zeros((x.shape[0], self.latent_dim, self.latent_dim)).to(x.device)
            cholesky_factors = params[:, self.latent_dim:]
            self.populate_cholesky(cholesky_factors)
            z = self.posterior.sample(mu, self.cholesky)
            log_qz_u = self.posterior.log_pdf_full(z, mu, self.cholesky)

        if self.prior.name == 'beta' or self.prior.name == 'uniform':
            determ = torch.log(1 / (torch.sigmoid(z) * (1 - torch.sigmoid(z)))).sum(1)
            log_qz_u += determ
            z = torch.sigmoid(z)

        x_recon = self.decoder(z)
        return x_recon, z, log_qz_u
