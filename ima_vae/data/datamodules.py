from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from ima_vae.data.data_generators import ConditionalDataset
from ima_vae.data.data_generators import gen_data


class IMADataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 64, orthog: bool = False, mobius: bool = False,
                 linear: bool = False, latent_dim: int = 5, n_segments: int = 1, n_layers: int = 1, n_obs: int = 60e3,
                 seed: int = 1, train_ratio: float = .7, val_ratio: float = 0.2, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        print(f"{self.hparams.batch_size=}")

    def setup(self, stage: Optional[str] = None):

        # generate data
        n_obs_per_seg = int(self.hparams.n_obs / self.hparams.n_segments)

        obs, labels, sources = gen_data(Ncomp=self.hparams.latent_dim, Nlayer=self.hparams.n_layers,
                                        Nsegment=self.hparams.n_segments, NsegmentObs=n_obs_per_seg,
                                        orthog=self.hparams.orthog, mobius=self.hparams.mobius, seed=self.hparams.seed,
                                        NonLin="none" if self.hparams.linear is True else 'lrelu')

        ima_full = ConditionalDataset(obs, labels, sources)

        # split
        train_len = int(self.hparams.train_ratio * self.hparams.n_obs)
        val_len = int(self.hparams.val_ratio * self.hparams.n_obs)
        test_len = int(self.hparams.n_obs - train_len - val_len)

        self.ima_train, self.ima_val, self.ima_test_pred = random_split(ima_full, [train_len, val_len, test_len])

    def train_dataloader(self):
        return DataLoader(self.ima_train, shuffle=True, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.ima_val, shuffle=True, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.ima_test_pred, shuffle=True, batch_size=self.hparams.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.ima_test_pred, shuffle=True, batch_size=self.hparams.batch_size)

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        ...
