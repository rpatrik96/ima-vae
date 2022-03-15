from argparse import Namespace

import hydra
import pytest
from hydra import compose, initialize
from pytorch_lightning import seed_everything


@pytest.fixture(autouse=True)
def args():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../configs", job_name="test_app")

    cfg = compose(
        config_name="trainer",
    )

    seed_everything(cfg.seed_everything)

    return Namespace(**cfg)
