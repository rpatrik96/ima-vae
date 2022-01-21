from typing import get_args

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import wandb
import torch
import torch.nn as nn
from jax import jacfwd
from jax import numpy as jnp
from torch.autograd.functional import jacobian

import ima_vae.metrics
from ima.ima.metrics import jacobian_amari_distance, observed_data_likelihood
from ima_vae.models.ivae.ivae_core import ActivationType
from ima_vae.models.ivae.ivae_core import iVAE


# from disentanglement_lib.evaluation.metrics import mig, unsupervised_metrics, beta_vae, dci, factor_vae, irs, modularity_explicitness, unified_scores


def calc_jacobian(model: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Jacobian
    :param model: the model to calculate the Jacobian of
    :param latents: the inputs for evaluating the model
    :return: n_out x n_in
    """

    # set to eval mode but remember original state
    in_training: bool = model.training
    model.eval()  # otherwise we will get 0 gradients

    J = jacobian(lambda x: model.forward(x).sum(dim=0), latents).permute(1, 0, 2).abs().mean(0)

    # set back to original mode
    if in_training is True:
        model.train()

    return J


class IMAModule(pl.LightningModule):

    def __init__(self, device: str, activation: ActivationType, latent_dim: int = 2, n_segments: int = 1,
                 n_layers: int = 1, lr: float = 1e-4, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.model: iVAE = iVAE(latent_dim=latent_dim,
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
        elbo, z_est, rec_loss, kl_loss, latent_stat = self.model.elbo(obs, labels)
        neg_elbo = elbo.mul(-1)

        self._log_metrics(kl_loss, neg_elbo, rec_loss, latent_stat, "Metrics/train")

        return neg_elbo

    def _log_metrics(self, kl_loss, neg_elbo, rec_loss, latent_stat, panel_name):
        self.log(f"{panel_name}/neg_elbo", neg_elbo)
        self.log(f"{panel_name}/rec_loss", rec_loss)
        self.log(f"{panel_name}/kl_loss", kl_loss)
        self.log(f"{panel_name}/latent_statistics", latent_stat)

    # def log_disentanglement_metrics(self):
    #
    #     mig.compute_mig()
    #     unified_scores.compute_unified_scores()
    #     unsupervised_metrics.unsupervised_metrics()
    #     dci.compute_dci()
    #     irs.compute_irs()
    #     modularity_explicitness.compute_modularity_explicitness()
    #     beta_vae.compute_beta_vae_sklearn()
    #     factor_vae.compute_factor_vae()

    def validation_step(self, batch, batch_idx):
        obs, labels, sources = batch
        elbo, latent, rec_loss, kl_loss, latent_stat, reconstruction = self.model.elbo(obs, labels,
                                                                                       reconstruction=True)

        neg_elbo = elbo.mul(-1)

        panel_name = "Metrics/val"
        self._log_metrics(kl_loss, neg_elbo, rec_loss, latent_stat, panel_name)
        self._log_mcc(latent, sources, panel_name)
        self._log_cima(latent, panel_name)
        self._log_amari_dist(obs, panel_name)
        self._log_true_data_likelihood(obs, panel_name)
        self._log_latents(latent, panel_name)
        self._log_reconstruction(obs, reconstruction, panel_name)

        return neg_elbo

    def _log_reconstruction(self, obs, rec, panel_name, max_img_num: int = 5):
        if rec is not None and self.hparams.use_wandb is True and self.hparams.log_reconstruction is True and isinstance(
                self.logger,
                pl.loggers.wandb.WandbLogger) is True:
            wandb_logger = self.logger.experiment
            # not images

            if len(rec.shape) == 2:
                table = wandb.Table(columns=[f"dim={i}" for i in range(self.hparams.latent_dim)])
                imgs = []
                for i in range(self.hparams.latent_dim):
                    imgs.append(wandb.Image(plt.scatter(obs[:, i], rec[:, i], label=[f"obs_{i}", f"rec_{i}"])))

                table.add_data(*imgs)

            # images
            else:
                table = wandb.Table(columns=['Observations', 'Reconstruction'])
                for i in range(max_img_num):
                    table.add_data(wandb.Image(obs[i, :]), wandb.Image(rec[i, :]))

            wandb_logger.log({f"{panel_name}/reconstructions": table})

    def _log_mcc(self, estimated_factors, sources, panel_name, log=True):
        mat, _, _ = ima_vae.metrics.mcc.correlation(sources.permute(1, 0).numpy(),
                                                    estimated_factors.permute(1, 0).numpy(), method='Pearson')
        mcc = np.mean(np.abs(np.diag(mat)))
        if log is True:
            self.log(f"{panel_name}/mcc", mcc)

        return mcc

    def _log_cima(self, latent, panel_name, log=True):
        unmix_jacobian = calc_jacobian(self.model.decoder, latent)
        cima = cima_kl_diagonality(unmix_jacobian)

        if log is True:
            self.log(f"{panel_name}/cima", cima)

        return cima

    def _log_true_data_likelihood(self, obs, panel_name, log=True):
        # todo: setup the base_log_pdf
        true_data_likelihood = observed_data_likelihood(obs, lambda x: jnp.stack(
            [jacfwd(self.trainer.datamodule.unmixing)(jnp.array(xx)) for xx in x]))

        if log is True:
            self.log(f"{panel_name}/true_data_likelihood", true_data_likelihood.mean().tolist())

        return true_data_likelihood

    def _log_amari_dist(self, obs, panel_name, log=True):

        J = lambda xx: jnp.array(
            jacobian(lambda x: self.model.decoder.forward(x).sum(dim=0), torch.Tensor(xx.tolist())).permute(1, 0, 2))

        amari_dist = jacobian_amari_distance(jnp.array(obs), J, lambda x: jnp.stack(
            [jacfwd(self.trainer.datamodule.mixing)(xx) for xx in x]),
                                             self.trainer.datamodule.unmixing)

        if log is True:
            self.log(f"{panel_name}/amari_dist", amari_dist.tolist())

        return amari_dist

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("IMA")

        parser.add_argument("--activation", type=str, choices=get_args(ActivationType), default='none')
        parser.add_argument("--device", type=torch.device,
                            default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # parser.add_argument('--latent_dim', type=int, default=2, help='Latent/data dimension')
        # parser.add_argument('--n_segments', type=int, default=40, help='Number of clusters in latent space')
        # parser.add_argument('--n_layers', type=int, default=1, help='Number of layers in mixing')
        parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
        parser.add_argument('--log-latents', action='store_true', help="Log the latents pairwise")
        parser.add_argument('--log-reconstruction', action='store_true', help="Log the reconstructions")

        return parent_parser

    def _log_latents(self, latent, panel_name):

        if self.hparams.use_wandb is True and self.hparams.log_latents is True and isinstance(self.logger,
                                                                                              pl.loggers.wandb.WandbLogger) is True:

            wandb_logger = self.logger.experiment
            table = wandb.Table(columns=["Idx"] + [f"latent_{i}" for i in range(self.hparams.latent_dim)])
            for i in range(self.hparams.latent_dim - 1):
                imgs = [i]
                for j in range(i, self.hparams.latent_dim):
                    imgs.append(
                        wandb.Image(plt.scatter(latent[:, i], latent[:, j], label=[f"latent_{i}", f"latent_{j}"])))

                imgs += ([None] * (i))

                table.add_data(*imgs)

            wandb_logger.log({f"{panel_name}/latents": table})


def cima_kl_diagonality(jacobian):
    """
    Calculates the IMA constrast. Able to handle jax and Pytorch objects as well

    :param jacobian: jacobian matrix (jax or Pytorch)
    :return:
    """
    jacobian_t_jacobian = jacobian.T @ jacobian

    lib = torch if type(jacobian) is torch.Tensor else jnp

    return 0.5 * (lib.linalg.slogdet(lib.diag(lib.diag(jacobian_t_jacobian)))[1] -
                  lib.linalg.slogdet(jacobian_t_jacobian)[1])
