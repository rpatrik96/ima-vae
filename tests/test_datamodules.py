import torch

from ima_vae.data.datamodules import IMADataModule


def test_ima_data_dims(args):
    module = IMADataModule.from_argparse_args(args)

    module.setup()
    obs, labels, sources = next(iter(module.train_dataloader()))
    assert obs.shape == torch.Size([module.hparams.batch_size, module.hparams.latent_dim])
    assert labels.shape == torch.Size([module.hparams.batch_size, module.hparams.n_segments])
    assert sources.shape == torch.Size([module.hparams.batch_size, module.hparams.latent_dim])
