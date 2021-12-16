import numpy as np
import pytorch_lightning as pl
import torch

import ima_vae.metrics
from ima_vae.models.ivae.ivae_core import ActivationType
from ima_vae.models.ivae.ivae_core import iVAE

from typing import get_args



class IMAModule(pl.LightningModule):

    def __init__(self, device: str, activation: ActivationType, latent_dim: int = 2, n_segments: int = 40,
                 n_layers: int = 3, lr: float = 1e-4, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.model = iVAE(latent_dim=latent_dim,
                          data_dim=latent_dim,
                          n_segments=n_segments,
                          n_layers=n_layers,
                          hidden_dim=latent_dim * 10,
                          activation=activation,
                          device=device)

    def forward(self, obs, labels):
        # in lightning, forward defines the prediction/inference actions
        return self.model(obs, labels)

    def training_step(self, batch, batch_idx):
        obs, labels, sources = batch
        elbo, z_est, rec_loss, kl_loss = self.model.elbo(obs, labels)
        loss = elbo.mul(-1)

        self.log_metrics(kl_loss, loss, rec_loss, "Metrics/train")

        return loss

    def log_metrics(self, kl_loss, loss, rec_loss, panel_name):
        self.log(f"{panel_name}/loss", loss)
        self.log(f"{panel_name}/rec_loss", rec_loss)
        self.log(f"{panel_name}/kl_loss", kl_loss)

    def validation_step(self, batch, batch_idx):
        obs, labels, sources = batch
        elbo, estimated_factors, rec_loss, kl_loss  = self.model.elbo(obs, labels)

        mat, _, _ = ima_vae.metrics.mcc.correlation(sources.permute(1, 0).numpy(),
                                                    estimated_factors.permute(1, 0).numpy(), method='Pearson')
        mcc = np.mean(np.abs(np.diag(mat)))

        val_loss = elbo.mul(-1)


        panel_name = "Metrics/val"
        self.log_metrics(kl_loss, val_loss, rec_loss, panel_name)

        self.log(f"{panel_name}/mcc", mcc)

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("IMA")

        parser.add_argument("--activation", type=str, choices=get_args(ActivationType), default='none')
        parser.add_argument("--device", type=torch.device, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # parser.add_argument('--latent_dim', type=int, default=2, help='Latent/data dimension')
        # parser.add_argument('--n_segments', type=int, default=40, help='Number of clusters in latent space')
        # parser.add_argument('--n_layers', type=int, default=1, help='Number of layers in mixing')
        parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')

        return parent_parser
