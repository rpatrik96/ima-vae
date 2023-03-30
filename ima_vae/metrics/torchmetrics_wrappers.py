from torchmetrics import Metric, MeanMetric
import torch
from .mcc import correlation
import numpy as np
from .cima import cima_kl_diagonality
from .amari import amari_distance
from .conformal import conformal_contrast, col_norm_var, col_norms


class MCC(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("mcc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, sources: torch.Tensor, estimated_factors: torch.Tensor):
        assert sources.shape == estimated_factors.shape

        mat, _, _ = correlation(
            sources.permute(1, 0).cpu().numpy(),
            estimated_factors.permute(1, 0).cpu().numpy(),
            method="Pearson",
        )

        self.mcc += np.mean(np.abs(np.diag(mat)))
        self.total += 1

    def compute(self):
        return self.mcc.float() / self.total


class CIMA(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("cima", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, unmix_jacobian: torch.Tensor):
        self.cima += cima_kl_diagonality(unmix_jacobian)
        self.total += 1

    def compute(self):
        return self.cima.float() / self.total


class AmariDistance(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("amari", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mix_jacobian: torch.Tensor, unmix_jacobian: torch.Tensor):
        self.amari += amari_distance(mix_jacobian, unmix_jacobian)
        self.total += 1

    def compute(self):
        return self.amari.float() / self.total


class ConformalContrast(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "conformal_contrast", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, unmix_jacobian: torch.Tensor):
        self.conformal_contrast += conformal_contrast(unmix_jacobian)
        self.total += 1

    def compute(self):
        return self.conformal_contrast.float() / self.total


class ColumnNormVariance(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("col_norm_var", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, unmix_jacobian: torch.Tensor):
        self.col_norm_var += col_norm_var(unmix_jacobian)
        self.total += 1

    def compute(self):
        return self.col_norm_var.float() / self.total


class ColumnNorm(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("col_norms", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, unmix_jacobian: torch.Tensor):
        self.col_norms += col_norms(unmix_jacobian)
        self.total += 1

    def compute(self):
        return self.col_norms.float() / self.total


class VAEMetrics(object):
    def __init__(self):
        self.neg_elbo = MeanMetric()
        self.rec_loss = MeanMetric()
        self.kl_loss = MeanMetric()

    def update(self, neg_elbo, rec_loss, kl_loss):
        self.neg_elbo.update(neg_elbo)
        self.rec_loss.update(rec_loss)
        self.kl_loss.update(kl_loss)

    def compute(self):
        return self.neg_elbo.compute(), self.rec_loss.compute(), self.kl_loss.compute()


class LatentMetrics(object):
    def __init__(self, latent_dim: int):
        self.latent_dim = latent_dim
        self.means = {f"latent_mean_{i}": MeanMetric() for i in range(self.latent_dim)}
        self.variances = {
            f"latent_variance_{i}": MeanMetric() for i in range(self.latent_dim)
        }
        self.variance_of_means = MeanMetric()

    def update(self, latent_stats: dict):
        for key, val in latent_stats.items():
            if "latent_mean_variance" in key:
                self.variance_of_means.compute(val)
            elif "latent_mean" in key:
                self.means[key].update(val)
            elif "latent_variance" in key:
                self.variances[key].update(val)

    def compute(self):
        means = []
        variances = []
        for mean, var in zip(self.means.values(), self.variances.values()):
            means.append(mean.compute())
            variances.append(var.compute())

        return means, variances, self.variance_of_means.compute()
