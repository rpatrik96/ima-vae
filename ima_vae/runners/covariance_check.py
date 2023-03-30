from os.path import abspath, dirname, join

import torch

from ima_vae.data.datamodules import IMADataModule
from ima_vae.metrics.conformal import frobenius_diagonality
from ima_vae.runners.runner import IMAModule

if __name__ == "__main__":
    # source: https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing.html#checkpoint-loading
    FOLDER = dirname(dirname(dirname(abspath(__file__))))
    SUBFOLDER = "dsprites/7out3y24"
    DIR = join(FOLDER, SUBFOLDER)
    FILE = "epoch=499-step=41499.ckpt"
    PATH = join(DIR, FILE)

    # To load a model along with its weights and hyperparameters use the following method
    model = IMAModule.load_from_checkpoint(
        PATH, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    dm = IMADataModule(**model.hparams)
    dm.setup()
    num_batches = 100

    encoding_means = []
    sampled_latents = []
    # mean_reconstructions = []
    # sampled_reconstructions = []

    for idx, batch in zip(range(num_batches), dm.train_dataloader()):
        obs, labels, sources = batch
        (
            encoding_mean,
            encoding_logvar,
            latents,
            reconstructions,
            log_qz_xu,
        ) = model.model.forward(obs)

        # decoded_mean_latents = model.model.decode(encoding_mean)

        encoding_means.append(encoding_mean)
        sampled_latents.append(latents)
        # mean_reconstructions.append(decoded_mean_latents)
        # sampled_reconstructions.append(reconstructions)

    encoding_means = torch.cat(encoding_means).T
    sampled_latents = torch.cat(sampled_latents).T
    # mean_reconstructions = torch.cat(mean_reconstructions).reshape( num_batches*dm.hparams.batch_size,-1).T
    # sampled_reconstructions = torch.cat(sampled_reconstructions).reshape( num_batches*dm.hparams.batch_size,-1).T

    encoding_means_cov = torch.cov(encoding_means)
    sampled_latents_cov = torch.cov(sampled_latents)
    # mean_reconstructions_cov = torch.cov(mean_reconstructions)
    # sampled_reconstructions_cov = torch.cov(sampled_reconstructions)

    print(f"{encoding_means_cov=}")
    print(
        f"Frobenius diagonality of encoding_means_cov={frobenius_diagonality(encoding_means_cov)}"
    )
    print(f"{sampled_latents_cov=}")
    print(
        f"Frobenius diagonality of sampled_latents_cov={frobenius_diagonality(sampled_latents_cov)}"
    )
    # print(f"Frobenius diagonality of mean_reconstructions_cov={frobenius_diagonality(mean_reconstructions_cov)}")
    # print(f"Frobenius diagonality of sampled_reconstructions_cov={frobenius_diagonality(sampled_reconstructions_cov)}")
