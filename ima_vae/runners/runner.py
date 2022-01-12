from typing import get_args

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from jax import numpy as jnp
from torch.autograd.functional import jacobian

import ima_vae.metrics
from ima_vae.models.ivae.ivae_core import ActivationType
from ima_vae.models.ivae.ivae_core import iVAE


# from disentanglement_lib.evaluation.metrics import mig, unsupervised_metrics, beta_vae, dci, factor_vae, irs, modularity_explicitness, unified_scores


def calc_jacobian(model: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Jacobian more efficiently than ` torch.autograd.functional.jacobian`
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
        elbo, z_est, rec_loss, kl_loss = self.model.elbo(obs, labels)
        neg_elbo = elbo.mul(-1)

        self.log_metrics(kl_loss, neg_elbo, rec_loss, "Metrics/train")

        return neg_elbo

    def log_metrics(self, kl_loss, neg_elbo, rec_loss, panel_name):
        self.log(f"{panel_name}/neg_elbo", neg_elbo)
        self.log(f"{panel_name}/rec_loss", rec_loss)
        self.log(f"{panel_name}/kl_loss", kl_loss)

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
        elbo, estimated_factors, rec_loss, kl_loss = self.model.elbo(obs, labels)

        neg_elbo = elbo.mul(-1)

        panel_name = "Metrics/val"
        self.log_metrics(kl_loss, neg_elbo, rec_loss, panel_name)

        self._mcc_calc_and_log(estimated_factors, sources, panel_name)
        self._cima_calc_and_log(obs, labels, panel_name)

        # calculate Amari distance
        # amari_distance()
        # jacobian_amari_distance()

        return neg_elbo

    def _mcc_calc_and_log(self, estimated_factors, sources, panel_name, log=True):
        mat, _, _ = ima_vae.metrics.mcc.correlation(sources.permute(1, 0).numpy(),
                                                    estimated_factors.permute(1, 0).numpy(), method='Pearson')
        mcc = np.mean(np.abs(np.diag(mat)))
        if log is True:
            self.log(f"{panel_name}/mcc", mcc)

        return mcc

    def _cima_calc_and_log(self, obs, labels, panel_name, log=True):
        decoder_params, encoder_params, latent, prior_params = self.model(obs, labels)
        jacobian = calc_jacobian(self.model.decoder, latent)
        cima = cima_kl_diagonality(jacobian)

        if log is True:
            self.log(f"{panel_name}/cima", cima)

        return cima

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

        return parent_parser


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
