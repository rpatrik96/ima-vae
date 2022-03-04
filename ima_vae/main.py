from argparse import Namespace

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from ima_vae.args import parse_args
from ima_vae.data.datamodules import IMADataModule
from ima_vae.runners.runner import IMAModule


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # install the package

    seed_everything(cfg.seed_everything)

    trainer = Trainer.from_argparse_args(Namespace(**cfg))
    model = IMAModule(**OmegaConf.to_container(cfg.model))
    dm = IMADataModule.from_argparse_args(Namespace(**cfg.data))

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
    # parser = parse_args()
    #
    # # add model specific args
    # parser = IMAModule.add_model_specific_args(parser)
    # # add data specific args
    # parser = IMADataModule.add_argparse_args(parser)
    # # add all the available trainer options to argparse
    # # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    # parser = Trainer.add_argparse_args(parser)
    # args = parser.parse_args()
    #
    # args.mobius = True
    #
    # seed_everything(args.seed)
    #
    # if args.wandb:
    #     args.logger = WandbLogger(
    #         entity="ima-vae", project=args.project, notes=args.notes, tags=args.tags
    #     )
    #
    # if args.dataset == "image":
    #     args.latent_dim = 4
    #
    # dict_args = vars(args)
    #
    # # init the trainer like this
    # trainer = Trainer.from_argparse_args(args)
    #
    # # init the model with all the key-value pairs
    # model = IMAModule(**dict_args)
    #
    # if args.wandb is True:
    #     args.logger.watch(model, log="all", log_freq=250)
    #
    # # datamodule
    # dm = IMADataModule.from_argparse_args(args)
    #
    # # fit
    # trainer.fit(model, datamodule=dm)
