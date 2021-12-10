from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from ima_vae.data.datamodules import IMADataModule
from ima_vae.runners.runner import IMAModule


def test_ima_args(args):
    # add model specific args
    runner = IMAModule(**vars(args))

    # if the code reaches this point, then all required args are specified
    pass


def test_training_with_wandb_logging(args):
    args.fast_dev_run = True
    args.logger = WandbLogger(project="test", entity="ima-vae", offline=True)

    dict_args = vars(args)

    # init the trainer like this
    trainer = Trainer.from_argparse_args(args)

    # init the model with all the key-value pairs
    model = IMAModule(**dict_args)

    # datamodule
    dm = IMADataModule.from_argparse_args(args)

    # fit
    trainer.fit(model, datamodule=dm)
