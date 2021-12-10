from pytorch_lightning import Trainer

from ima_vae.data.datamodules import IMADataModule
from ima_vae.runners.lightning_runner import IMAModule


def test_ima_args(args):
    # add model specific args
    runner = IMAModule(**vars(args))

    # if the code reaches this point, then all required args are specified
    pass


def test_training(args):
    args.fast_dev_run = True
    dict_args = vars(args)

    # init the trainer like this
    trainer = Trainer.from_argparse_args(args)

    # init the model with all the key-value pairs
    model = IMAModule(**dict_args)

    # datamodule
    dm = IMADataModule.from_argparse_args(args)

    # fit
    trainer.fit(model, datamodule=dm)
