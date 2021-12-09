import pytorch_lightning as pl

from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from ima_vae.models.ivae.ivae_core import iVAE


class IMAModule(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.model = iVAE(latent_dim=latent_dim,
                     data_dim=data_dim,
                     aux_dim=aux_dim,
                     n_layers=n_layers,
                     hidden_dim=hidden_dim,
                     activation=activation,
                     device=device)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # embedding = self.encoder(x)
        # return embedding
        pass

    def training_step(self, batch, batch_idx):
        obs, labels, sources = batch
        elbo, z_est = self.model.elbo(obs, labels)
        loss = elbo.mul(-1)

        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
