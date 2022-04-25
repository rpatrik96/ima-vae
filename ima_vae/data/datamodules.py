from os.path import dirname, abspath
from typing import Optional

import pytorch_lightning as pl
import torchvision.transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from ima_vae.data.data_generators import gen_synth_dataset
from ima_vae.data.dataset import ConditionalDataset
from ima_vae.data.utils import DatasetType, load_sprites


class IMADataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = dirname(abspath(__file__)),
        batch_size: int = 64,
        orthog: bool = False,
        mobius: bool = True,
        linear: bool = False,
        latent_dim: int = 2,
        n_segments: int = 1,
        mixing_layers: int = 1,
        n_obs: int = int(60e3),
        seed: int = 1,
        n_classes: int = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        dataset: DatasetType = "synth",
        synth_source="uniform",
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        prior_var: float = 1.0,
        prior_mean: float = 0.0,
        ar_flow: bool = False,
        projective: bool = False,
        affine: bool = False,
        deltah: int = 0,
        deltas: int = 0,
        deltav: int = 0,
        angle: bool = False,
        shape: bool = False,
        **kwargs,
    ):
        """

        :param angle: angle flag for dSprites
        :param shape: shape flag for dSprites
        :param deltah: Disturbance in the Hue channel
        :param deltas: Disturbance in the Saturation channel
        :param deltav: Disturbance in the Value channel
        :param affine: flag to use affine transformation for image generation
        :param projective: flag to use projective transformation for image generation
        :param ar_flow: use ar_flow in the data generation process
        :param prior_alpha: beta prior alpha shape > 0
        :param prior_beta: beta prior beta shape > 0
        :param prior_mean: prior mean
        :param prior_var: prior variance
        :param data_dir: data directory
        :param batch_size: batch size
        :param orthog: orthogonality flag for mixing
        :param mobius: flag for the Moebius transform
        :param linear: flag for activation linearity
        :param latent_dim: latent dimension
        :param n_segments: number of segments (for iVAE-like conditional data)
        :param mixing_layers: number of layers (if mixing is done with an MLP)
        :param n_obs: number of observations
        :param seed: seed
        :param n_classes: number of classes
        :param train_ratio: train ratio
        :param val_ratio: validation ratio
        :param dataset: dataset specifier, can be any of ["synth", "image"]
        :param synth_source: source distribution for synthetic data, can be ["uniform", "gaussian", "laplace", "beta"]
        :param kwargs:
        """
        super().__init__()

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        # generate data

        if self.hparams.dataset == "image":
            transform = torchvision.transforms.ToTensor()
            (
                labels,
                obs,
                sources,
                self.mixing,
                self.unmixing,
                self.discrete_list,
            ) = load_sprites(
                self.hparams.n_obs,
                self.hparams.n_classes,
                self.hparams.projective,
                self.hparams.affine,
                self.hparams.deltah,
                self.hparams.deltas,
                self.hparams.deltav,
                self.hparams.angle,
                self.hparams.shape,
            )
        elif self.hparams.dataset == "synth":
            transform = None

            n_obs_per_seg = int(self.hparams.n_obs / self.hparams.n_segments)

            (
                obs,
                labels,
                sources,
                self.mixing,
                self.unmixing,
                self.discrete_list,
            ) = gen_synth_dataset.gen_data(
                num_dim=self.hparams.latent_dim,
                num_layer=self.hparams.mixing_layers,
                num_segment=self.hparams.n_segments,
                num_segment_obs=n_obs_per_seg,
                orthog=self.hparams.orthog,
                seed=self.hparams.seed,
                nonlin="none" if self.hparams.linear is True else "lrelu",
                source=self.hparams.synth_source,
                mobius=self.hparams.mobius,
                alpha_shape=self.hparams.prior_alpha,
                beta_shape=self.hparams.prior_beta,
                mean=self.hparams.prior_mean,
                var=self.hparams.prior_var,
                ar_flow=self.hparams.ar_flow,
            )
        else:
            raise ValueError

        if self.mixing is None:
            print(f"Mixing is unknown, a reduced set of metrics is calculated!")
        if self.unmixing is None:
            print(f"Unmixing is unknown, a reduced set of metrics is calculated!")

        ima_full = ConditionalDataset(obs, labels, sources, transform=transform)

        # split
        train_len = int(self.hparams.train_ratio * self.hparams.n_obs)
        val_len = int(self.hparams.val_ratio * self.hparams.n_obs)
        test_len = int(self.hparams.n_obs - train_len - val_len)

        self.ima_train, self.ima_val, self.ima_test_pred = random_split(
            ima_full, [train_len, val_len, test_len]
        )

    def train_dataloader(self):
        return DataLoader(
            self.ima_train, shuffle=True, batch_size=self.hparams.batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            self.ima_val, shuffle=False, batch_size=self.hparams.batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            self.ima_test_pred, shuffle=False, batch_size=self.hparams.batch_size
        )

    def predict_dataloader(self):
        return DataLoader(
            self.ima_test_pred, shuffle=False, batch_size=self.hparams.batch_size
        )

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        ...
