from typing import get_args, List

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
from ima_vae.metrics.conformal import conformal_contrast, col_norm_var
import ima_vae.metrics
from ima.ima.metrics import jacobian_amari_distance, observed_data_likelihood
from ima_vae.data.utils import DatasetType
from ima_vae.metrics.cima import cima_kl_diagonality
from ima_vae.metrics.mig import compute_mig_with_discrete_factors
from ima_vae.models.ivae import iVAE
from ima_vae.models.utils import ActivationType
# from disentanglement_lib.evaluation.metrics import mig, unsupervised_metrics, beta_vae, dci, factor_vae, irs, modularity_explicitness, unified_scores
from ima_vae.utils import calc_jacobian


class IMAModule(pl.LightningModule):

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 activation: ActivationType = 'none', latent_dim: int = 2, n_segments: int = 1,
                 n_layers: int = 1, lr: float = 1e-4, n_classes: int = 1, dataset: DatasetType = 'synth',
                 log_latents: bool = False, log_reconstruction: bool = False, **kwargs):
        """

        :param device: device to run on
        :param activation: activation function, any on 'lrelu', 'sigmoid', 'none'
        :param latent_dim: dimension of the latent space
        :param n_segments: number of segments (for iVAE-like data, currently unused)
        :param n_layers: number of layers
        :param lr: learning rate
        :param n_classes: number of classes
        :param log_latents: flag to log latents
        :param log_reconstruction: flag to log reconstructions
        :param kwargs:
        """
        super().__init__()

        self.save_hyperparameters()

        self.model: iVAE = iVAE(latent_dim=latent_dim, data_dim=latent_dim, n_segments=n_segments, n_classes=n_classes,
                                n_layers=n_layers, hidden_dim=latent_dim * 10, activation=activation, device=device,
                                dataset=self.hparams.dataset)

        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            self.logger.watch(self.model, log="all", log_freq=250)

    def forward(self, obs, labels):
        # in lightning, forward defines the prediction/inference actions
        return self.model(obs, labels)

    def training_step(self, batch, batch_idx):
        obs, labels, sources = batch
        elbo, z_est, rec_loss, kl_loss, latent_stat, _ = self.model.elbo(obs, labels)
        neg_elbo = elbo.mul(-1)

        self._log_metrics(kl_loss, neg_elbo, rec_loss, latent_stat, "Metrics/train")

        return neg_elbo

    def _log_metrics(self, kl_loss, neg_elbo, rec_loss, latent_stat, panel_name):
        self.log(f"{panel_name}/neg_elbo", neg_elbo)
        self.log(f"{panel_name}/rec_loss", rec_loss)
        self.log(f"{panel_name}/kl_loss", kl_loss)
        self.log(f"{panel_name}/latent_statistics", latent_stat)

    def _log_disentanglement_metrics(self, sources, predicted_latents, discrete_list: List[bool], panel_name,
                                     continuous_factors: bool = True, train_split=0.8):

        pass

        """
            mus: mean latents
            ys: generating factors
        """

        num_samples = predicted_latents.shape[0]
        num_train = int(train_split * num_samples)

        mus_train, mus_test = predicted_latents[:num_train, :], predicted_latents[num_train:, :]
        ys_train, ys_test = sources[:num_train, :], sources[num_train:, :]

        sap: dict = _compute_sap(mus_train, ys_train, mus_test, ys_test, continuous_factors)
        self.log(f"{panel_name}/sap", sap)

        # uses train-val-test splits of 0.8-0.1-0.1
        mig: dict = compute_mig_with_discrete_factors(predicted_latents, sources, discrete_list)
        self.log(f"{panel_name}/mig", mig)

        if continuous_factors is False:
            dci: dict = _compute_dci(mus_train, ys_train, mus_test, ys_test)
            self.log(f"{panel_name}/dci", dci)

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
        self._log_disentanglement_metrics(sources, latent, self.trainer.datamodule.discrete_list, panel_name,
                                          continuous_factors=False in self.trainer.datamodule.discrete_list)

        return neg_elbo

    def _log_reconstruction(self, obs, rec, panel_name, max_img_num: int = 5):
        if rec is not None and self.hparams.log_reconstruction is True and isinstance(
                self.logger,
                pl.loggers.wandb.WandbLogger) is True:
            wandb_logger = self.logger.experiment
            # not images

            if len(obs.shape) == 2:
                table = wandb.Table(columns=[f"dim={i}" for i in range(self.hparams.latent_dim)])
                imgs = []
                for i in range(self.hparams.latent_dim):
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    imgs.append(wandb.Image(ax.scatter(obs[:, i], rec[:, i], label=[f"obs_{i}", f"rec_{i}"])))

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
            self.log(f"{panel_name}/conformal_contrast", conformal_contrast(unmix_jacobian))
            self.log(f"{panel_name}/col_norm_var", col_norm_var(unmix_jacobian))

        return cima

    def _log_true_data_likelihood(self, obs, panel_name, log=True):
        # todo: setup the base_log_pdf
        if self.trainer.datamodule.unmixing is not None:
            true_data_likelihood = observed_data_likelihood(obs, lambda x: jnp.stack(
                [jacfwd(self.trainer.datamodule.unmixing)(jnp.array(xx)) for xx in x]))

            if log is True:
                self.log(f"{panel_name}/true_data_likelihood", true_data_likelihood.mean().tolist())
        else:
            true_data_likelihood = None

        return true_data_likelihood

    def _log_amari_dist(self, obs, panel_name, log=True):

        if self.trainer.datamodule.mixing is not None or self.trainer.datamodule.unmixing is not None:
            J = lambda xx: jnp.array(
                jacobian(lambda x: self.model.decoder.forward(x).sum(dim=0), torch.Tensor(xx.tolist())).permute(1, 0,
                                                                                                                2))

            amari_dist = jacobian_amari_distance(jnp.array(obs), J, lambda x: jnp.stack(
                [jacfwd(self.trainer.datamodule.mixing)(xx) for xx in x]),
                                                 self.trainer.datamodule.unmixing)

            if log is True:
                self.log(f"{panel_name}/amari_dist", amari_dist.tolist())
        else:
            amari_dist = None

        return amari_dist

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("IMA")

        parser.add_argument("--activation", type=str, choices=get_args(ActivationType), default='none')
        parser.add_argument("--dataset", type=str, choices=get_args(DatasetType), default='synth')
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

        if self.logger is not None and self.hparams.log_latents is True and isinstance(self.logger,
                                                                                       pl.loggers.wandb.WandbLogger) is True:

            wandb_logger = self.logger.experiment
            table = wandb.Table(columns=["Idx"] + [f"latent_{i}" for i in range(self.hparams.latent_dim)])
            for row in range(self.hparams.latent_dim - 1):
                imgs = [row]
                imgs += ([None] * (row + 1))
                for col in range(row + 1, self.hparams.latent_dim):
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    imgs.append(
                        wandb.Image(
                            ax.scatter(latent[:, row], latent[:, col], label=[f"latent_{row}", f"latent_{col}"])))

                table.add_data(*imgs)

            wandb_logger.log({f"{panel_name}/latents": table})
