import pytest
from pytorch_lightning import Trainer, seed_everything

from ima_vae.args import parse_args
from ima_vae.data.datamodules import IMADataModule
from ima_vae.runners.runner import IMAModule


@pytest.fixture(autouse=True)
def args():
    parser = parse_args()
    # add model specific args
    parser = IMAModule.add_model_specific_args(parser)
    # add data specific args
    parser = IMADataModule.add_argparse_args(parser)
    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])

    # seed
    seed_everything(args.seed)

    return args
