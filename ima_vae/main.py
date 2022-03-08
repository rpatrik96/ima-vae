from argparse import Namespace

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from ima_vae.args import parse_args
from ima_vae.data.datamodules import IMADataModule
from ima_vae.runners.runner import IMAModule


@hydra.main(config_path="../configs", config_name="trainer")
def main(cfg: DictConfig):
    # install the package

    seed_everything(cfg.seed_everything)

    trainer = Trainer.from_argparse_args(Namespace(**cfg))
    model = IMAModule(**OmegaConf.to_container(cfg.model))
    dm = IMADataModule.from_argparse_args(Namespace(**cfg.data))

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
