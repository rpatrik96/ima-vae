from os.path import dirname, abspath
from typing import Optional

import pytorch_lightning as pl
import torchvision.transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from ima_vae.data.data_generators import gen_synth_dataset
from ima_vae.data.dataset import ConditionalDataset
from ima_vae.data.utils import load_sprites, DatasetType


class IMADataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = dirname(abspath(__file__)), batch_size: int = 64, orthog: bool = False,
                 mobius: bool = False, linear: bool = False, latent_dim: int = 5, n_segments: int = 1,
                 n_layers: int = 1, n_obs: int = 10e3, seed: int = 1, n_classes: int = 1, train_ratio: float = .7,
                 val_ratio: float = 0.2, dataset: DatasetType = "synth", **kwargs):
        super().__init__()

        self.save_hyperparameters()

        print(f"{self.hparams.batch_size=}")

    def setup(self, stage: Optional[str] = None):
        # generate data

        if self.hparams.dataset == 'image':
            transform = torchvision.transforms.ToTensor()
            labels, obs, sources, self.mixing, self.unmixing, self.discrete_list = load_sprites(self.hparams.n_obs, self.hparams.n_classes)
        elif self.hparams.dataset == 'synth':
            transform = None

            n_obs_per_seg = int(self.hparams.n_obs / self.hparams.n_segments)

            obs, labels, sources, self.mixing, self.unmixing, self.discrete_list = gen_synth_dataset.gen_data(
                num_dim=self.hparams.latent_dim,
                num_layer=self.hparams.n_layers,
                num_segment=self.hparams.n_segments,
                num_segment_obs=n_obs_per_seg,
                orthog=self.hparams.orthog,
                mobius=self.hparams.mobius,
                seed=self.hparams.seed,
                nonlin="none" if self.hparams.linear is True else 'lrelu')

        if self.mixing is None:
            print(f"Mixing is unknown, a reduced set of metrics is calculated!")
        if self.unmixing is None:
            print(f"Unmixing is unknown, a reduced set of metrics is calculated!")

        ima_full = ConditionalDataset(obs, labels, sources, transform=transform)

        # split
        train_len = int(self.hparams.train_ratio * self.hparams.n_obs)
        val_len = int(self.hparams.val_ratio * self.hparams.n_obs)
        test_len = int(self.hparams.n_obs - train_len - val_len)

        self.ima_train, self.ima_val, self.ima_test_pred = random_split(ima_full, [train_len, val_len, test_len])

    def train_dataloader(self):
        return DataLoader(self.ima_train, shuffle=True, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.ima_val, shuffle=False, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.ima_test_pred, shuffle=False, batch_size=self.hparams.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.ima_test_pred, shuffle=False, batch_size=self.hparams.batch_size)

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        ...
