from argparse import Namespace

import hydra
import torch.cuda
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything

from ima_vae.data.datamodules import IMADataModule
from ima_vae.runners.runner import IMAModule


@hydra.main(config_path="../configs", config_name="trainer")
def main(cfg: DictConfig):
    seed_everything(cfg.seed_everything)

    if torch.cuda.is_available() is False:
        cfg.trainer.gpus = 0

    trainer = Trainer.from_argparse_args(Namespace(**cfg.trainer))
    model = IMAModule(**OmegaConf.to_container(cfg.model))
    dm = IMADataModule.from_argparse_args(Namespace(**cfg.data))

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
