from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers.wandb import WandbLogger

from ima_vae.data.datamodules import IMADataModule
from ima_vae.runners.runner import IMAModule


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):

        parser.add_argument("--notes", type=str, default=None, help="Notes for the run on Weights and Biases")
        # todo: process notes based on args in before_instantiate_classes
        parser.add_argument("--tags", type=str,
                            nargs="*",  # 0 or more values expected => creates a list
                            default=None, help="Tags for the run on Weights and Biases")

        parser.link_arguments("data.latent_dim", "model.latent_dim")




    def before_fit(self):
        if isinstance(self.trainer.logger, WandbLogger) is True:
            # required as the parser cannot parse the "-" symbol
            self.trainer.logger.__dict__['_wandb_init']['entity'] = 'ima-vae'

            # todo: maybe set run in the CLI to false and call watch before?
            self.trainer.logger.watch(self.model, log="all", log_freq=250)


cli = MyLightningCLI(IMAModule, IMADataModule, save_config_callback=None, run=True)
