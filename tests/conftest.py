from argparse import ArgumentParser

import pytest
from pytorch_lightning import Trainer

from ima_vae.data.datamodules import IMADataModule
from ima_vae.runners.lightning_runner import IMAModule


@pytest.fixture(autouse=True)
def args():
    parser = ArgumentParser()
    # add model specific args
    parser = IMAModule.add_model_specific_args(parser)
    # add data specific args
    parser = IMADataModule.add_argparse_args(parser)
    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return args