from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from ima_vae.data.datamodules import IMADataModule
from ima_vae.runners.runner import IMAModule
from ima_vae.utils import add_tags


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):

        parser.add_argument(
            "--notes",
            type=str,
            default=None,
            help="Notes for the run on Weights and Biases",
        )
        # todo: process notes based on args in before_instantiate_classes
        parser.add_argument(
            "--tags",
            type=str,
            nargs="*",  # 0 or more values expected => creates a list
            default=None,
            help="Tags for the run on Weights and Biases",
        )

        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults(
            {
                "early_stopping.monitor": "Metrics/val/neg_elbo",
                "early_stopping.mode": "min",
                "early_stopping.patience": 5,
            }
        )

        parser.link_arguments("model.latent_dim", "data.latent_dim")
        parser.link_arguments("model.prior_mean", "data.prior_mean")
        parser.link_arguments("model.prior_var", "data.prior_var")
        parser.link_arguments("model.prior_alpha", "data.prior_alpha")
        parser.link_arguments("model.prior_beta", "data.prior_beta")
        parser.link_arguments("model.dataset", "data.dataset")

    def before_instantiate_classes(self) -> None:

        if self.config[self.subcommand].model.dataset == "image":
            nfactors = (
                4
                + int(self.config[self.subcommand].data.shape)
                + int(self.config[self.subcommand].data.angle)
            )
            self.config[self.subcommand].model.latent_dim = self.config[
                self.subcommand
            ].data.latent_dim = nfactors

        self.config[self.subcommand].trainer.logger.init_args.tags = add_tags(
            self.config[self.subcommand]
        )
        import jax

        jax.config.update("jax_platform_name", "cpu")

    def before_fit(self):
        if isinstance(self.trainer.logger, WandbLogger) is True:
            # required as the parser cannot parse the "-" symbol
            self.trainer.logger.__dict__["_wandb_init"]["entity"] = "ima-vae"

            if self.config[self.subcommand].model.offline is True:
                self.trainer.logger.__dict__["_wandb_init"]["mode"] = "offline"
            else:
                self.trainer.logger.__dict__["_wandb_init"]["mode"] = "online"

            # todo: maybe set run in the CLI to false and call watch before?
            self.trainer.logger.watch(self.model, log="all", log_freq=250)


cli = MyLightningCLI(
    IMAModule,
    IMADataModule,
    save_config_callback=None,
    run=True,
    parser_kwargs={"parse_as_dict": False},
    trainer_defaults={
        "callbacks": [
            ModelCheckpoint(
                save_top_k=1,
                monitor="Metrics/val/neg_elbo",
                mode="min",
            ),
        ]
    },
)
