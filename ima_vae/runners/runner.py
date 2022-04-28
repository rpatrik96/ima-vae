from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from disent.metrics._dci import _compute_dci
from disent.metrics._sap import _compute_sap
from jax import jacfwd
from jax import numpy as jnp
from torch.autograd.functional import jacobian

import ima_vae.metrics
from ima.ima.metrics import jacobian_amari_distance, observed_data_likelihood
from ima_vae.data.utils import DatasetType
from ima_vae.metrics.cima import cima_kl_diagonality
from ima_vae.metrics.conformal import conformal_contrast, col_norm_var, col_norms
from ima_vae.metrics.amari import amari_distance
from ima_vae.metrics.mig import compute_mig_with_discrete_factors
from ima_vae.models.ivae import iVAE
from ima_vae.models.utils import ActivationType
from ima_vae.models.utils import PriorType
from ima_vae.utils import calc_jacobian


class IMAModule(pl.LightningModule):
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        activation: ActivationType = "lrelu",
        latent_dim: int = 2,
        n_segments: int = 1,
        n_layers: int = 2,
        lr: float = 1e-3,
        n_classes: int = 1,
        dataset: DatasetType = "synth",
        log_latents: bool = False,
        log_reconstruction: bool = False,
        prior: PriorType = "uniform",
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        prior_mean: float = 0.0,
        prior_var: float = 1.0,
        decoder_var=0.000001,
        fix_prior: bool = True,
        beta=1.0,
        diag_posterior: bool = True,
        **kwargs,
    ):
        """

        :param diag_posterior: choose a diagonal posterior
        :param beta: beta of the beta-VAE
        :param fix_prior: fix (and not learn) prior distribution
        :param decoder_var: decoder variance
        :param prior_mean: prior mean
        :param prior_var: prior variance
        :param prior_alpha: beta prior alpha shape > 0
        :param prior_beta: beta prior beta shape > 0
        :param device: device to run on
        :param activation: activation function, any on 'lrelu', 'sigmoid', 'none'
        :param latent_dim: dimension of the latent space
        :param n_segments: number of segments (for iVAE-like data, currently unused)
        :param n_layers: number of layers
        :param lr: learning rate
        :param n_classes: number of classes
        :param log_latents: flag to log latents
        :param log_reconstruction: flag to log reconstructions
        :param prior: prior distribution name as string
        :param kwargs:
        """
        super().__init__()

        self.save_hyperparameters()

        self.model: iVAE = iVAE(
            latent_dim=latent_dim,
            data_dim=latent_dim,
            n_segments=n_segments,
            n_classes=n_classes,
            n_layers=n_layers,
            activation=activation,
            device=device,
            prior=prior,
            dataset=self.hparams.dataset,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            prior_mean=prior_mean,
            prior_var=prior_var,
            decoder_var=decoder_var,
            fix_prior=fix_prior,
            beta=beta,
            diag_posterior=diag_posterior,
        )

        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            self.logger.watch(self.model, log="all", log_freq=250)

    def forward(self, obs, labels):
        # in lightning, forward defines the prediction/inference actions
        return self.model(obs, labels)

    def training_step(self, batch, batch_idx):
        obs, labels, sources = batch
        neg_elbo, z_est, rec_loss, kl_loss, latent_stat, _ = self.model.neg_elbo(
            obs, labels
        )

        panel_name = "Metrics/train"
        self._log_metrics(kl_loss, neg_elbo, rec_loss, latent_stat, panel_name)
        with torch.no_grad():
            self._log_mcc(z_est, sources, panel_name)

        return neg_elbo

    def _log_metrics(self, kl_loss, neg_elbo, rec_loss, latent_stat, panel_name):
        self.log(f"{panel_name}/neg_elbo", neg_elbo)
        self.log(f"{panel_name}/rec_loss", rec_loss)
        self.log(f"{panel_name}/kl_loss", kl_loss)
        self.log(f"{panel_name}/latent_statistics", latent_stat)

    def _log_disentanglement_metrics(
        self,
        sources,
        predicted_latents,
        discrete_list: List[bool],
        panel_name,
        continuous_factors: bool = True,
        train_split=0.8,
    ):

        pass

        """
            mus: mean latents
            ys: generating factors
        """

        num_samples = predicted_latents.shape[0]
        num_train = int(train_split * num_samples)

        mus_train, mus_test = (
            predicted_latents[:num_train, :],
            predicted_latents[num_train:, :],
        )
        ys_train, ys_test = sources[:num_train, :], sources[num_train:, :]

        sap: dict = _compute_sap(
            mus_train.cpu(),
            ys_train.cpu(),
            mus_test.cpu(),
            ys_test.cpu(),
            continuous_factors,
        )
        self.log(f"{panel_name}/sap", sap)

        # uses train-val-test splits of 0.8-0.1-0.1
        mig: dict = compute_mig_with_discrete_factors(
            predicted_latents.cpu(), sources.cpu(), discrete_list
        )
        self.log(f"{panel_name}/mig", mig)

        if continuous_factors is False:
            dci: dict = _compute_dci(mus_train, ys_train, mus_test, ys_test)
            self.log(f"{panel_name}/dci", dci)

    def validation_step(self, batch, batch_idx):
        obs, labels, sources = batch
        (
            neg_elbo,
            latent,
            rec_loss,
            kl_loss,
            latent_stat,
            reconstruction,
        ) = self.model.neg_elbo(obs, labels, reconstruction=True)

        panel_name = "Metrics/val"
        self._log_metrics(kl_loss, neg_elbo, rec_loss, latent_stat, panel_name)
        self._log_mcc(latent, sources, panel_name)

        if (
            (
                reduced_freq_flag := (
                    self.global_step > 2000
                    and self.global_step % 2000 == 0
                    or self.current_epoch == (self.trainer.max_epochs - 1)
                )
            )
            and self.hparams.dataset == "image"
        ) or self.hparams.dataset == "synth":
            self._log_cima(latent, panel_name)

        # todo: calc at the end of fit
        if reduced_freq_flag is True:
            self._log_amari_dist(obs, sources, panel_name)
            # self._log_true_data_likelihood(obs, panel_name) #uses jax
            self._log_latents(latent, panel_name)
            self._log_reconstruction(obs, reconstruction, panel_name)
            self._log_disentanglement_metrics(
                sources,
                latent,
                self.trainer.datamodule.discrete_list,
                panel_name,
                continuous_factors=False in self.trainer.datamodule.discrete_list,
            )

        return neg_elbo

    def _log_reconstruction(self, obs, rec, panel_name, max_img_num: int = 5):
        if (
            rec is not None
            and self.hparams.log_reconstruction is True
            and isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True
        ):
            wandb_logger = self.logger.experiment
            # not images

            if len(obs.shape) == 2:
                table = wandb.Table(
                    columns=[f"dim={i}" for i in range(self.hparams.latent_dim)]
                )
                imgs = []
                for i in range(self.hparams.latent_dim):
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    imgs.append(
                        wandb.Image(
                            ax.scatter(
                                obs[:, i], rec[:, i], label=[f"obs_{i}", f"rec_{i}"]
                            )
                        )
                    )

                table.add_data(*imgs)

            # images
            else:
                table = wandb.Table(columns=["Observations", "Reconstruction"])
                for i in range(max_img_num):
                    table.add_data(wandb.Image(obs[i, :]), wandb.Image(rec[i, :]))

            wandb_logger.log({f"{panel_name}/reconstructions": table})

    def _log_mcc(self, estimated_factors, sources, panel_name, log=True):
        mat, _, _ = ima_vae.metrics.mcc.correlation(
            sources.permute(1, 0).cpu().numpy(),
            estimated_factors.permute(1, 0).cpu().numpy(),
            method="Pearson",
        )
        mcc = np.mean(np.abs(np.diag(mat)))
        if log is True:
            self.log(f"{panel_name}/mcc", mcc, prog_bar=True)

        return mcc

    def _log_cima(self, latent, panel_name, log=True):
        unmix_jacobians = calc_jacobian(self.model.decoder, latent)
        unmix_jacobian = unmix_jacobians.mean(0)
        cima = cima_kl_diagonality(unmix_jacobian)

        if log is True:
            self.log(f"{panel_name}/cima", cima)
            self.log(
                f"{panel_name}/conformal_contrast", conformal_contrast(unmix_jacobian)
            )
            self.log(f"{panel_name}/col_norm_var", col_norm_var(unmix_jacobian))

            if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
                self.logger.experiment.log(
                    {f"{panel_name}/col_norms": col_norms(unmix_jacobian)}
                )

                # log max column norm for sample-wise jacobian

                sample_col_norms = torch.stack([col_norms(j) for j in unmix_jacobians])

                self.logger.experiment.log(
                    {
                        f"{panel_name}/sample_col_max_norms": sample_col_norms.max(0),
                        f"{panel_name}/sample_col_norms_var": sample_col_norms.var(0),
                    }
                )

        return cima

    def _log_true_data_likelihood(self, obs, panel_name, log=True):
        # todo: setup the base_log_pdf
        if self.trainer.datamodule.unmixing is not None:
            true_data_likelihood = observed_data_likelihood(
                obs,
                lambda x: jnp.stack(
                    [
                        jacfwd(self.trainer.datamodule.unmixing)(jnp.array(xx))
                        for xx in x
                    ]
                ),
            )

            if log is True:
                self.log(
                    f"{panel_name}/true_data_likelihood",
                    true_data_likelihood.mean().tolist(),
                )
        else:
            true_data_likelihood = None

        return true_data_likelihood

    def _log_amari_dist(self, obs, sources, panel_name, log=True):

        if (
            self.trainer.datamodule.mixing is not None
            and self.trainer.datamodule.unmixing is not None
        ):
            J_unmix = jacobian(
                lambda x: self.model.decoder.forward(x).sum(dim=0),
                obs,
            ).permute(1, 0, 2)
            J_mix = jacobian(
                lambda x: self.trainer.datamodule.mixing(x).sum(dim=0), sources
            ).permute(1, 0, 2)
            amari_dist = amari_distance(J_mix, J_unmix)
            if log is True:
                self.log(f"{panel_name}/amari_dist", amari_dist)
        else:
            amari_dist = None

        return amari_dist

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def _log_latents(self, latent, panel_name):

        if (
            self.logger is not None
            and self.hparams.log_latents is True
            and isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True
        ):

            wandb_logger = self.logger.experiment
            table = wandb.Table(
                columns=["Idx"]
                + [f"latent_{i}" for i in range(self.hparams.latent_dim)]
            )
            for row in range(self.hparams.latent_dim - 1):
                imgs = [row]
                imgs += [None] * (row + 1)
                for col in range(row + 1, self.hparams.latent_dim):
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    imgs.append(
                        wandb.Image(
                            ax.scatter(
                                latent[:, row],
                                latent[:, col],
                                label=[f"latent_{row}", f"latent_{col}"],
                            )
                        )
                    )

                table.add_data(*imgs)

            wandb_logger.log({f"{panel_name}/latents": table})
