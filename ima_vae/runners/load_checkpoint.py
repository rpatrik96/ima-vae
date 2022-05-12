from ima_vae.runners.runner import IMAModule
from pytorch_lightning import Trainer
from os.path import abspath, dirname, join
import torch


if __name__ == "__main__":
    # source: https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing.html#checkpoint-loading
    FOLDER = dirname(dirname(dirname(abspath(__file__))))
    FILE = "dsprites/7out3y24/epoch=499-step=41499.ckpt"
    PATH = join(FOLDER, FILE)

    # To load a model along with its weights and hyperparameters use the following method
    model = IMAModule.load_from_checkpoint(
        PATH, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # x = ...
    # model.eval()
    # y_hat = model(...)

    # If you don’t just want to load weights, but instead restore the full training, do the following:
    model = IMAModule()
    trainer = Trainer()

    # automatically restores model, epoch, step, LR schedulers, apex, etc...
    # trainer.fit(model, ckpt_path=PATH)
