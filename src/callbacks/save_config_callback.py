from typing import Optional
import os

from pytorch_lightning.cli import SaveConfigCallback
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import LightningModule

from utils import file_util, constants as C

class SaveAndLogConfigCallback(SaveConfigCallback):
    """A simple callback to log the config using the loggers of the trainer.
    """
    
    def setup(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: Optional[str] = None
    ) -> None:
        super().setup(trainer, pl_module, stage)
        if stage == 'fit' or 'fit' not in self.config[C.KEY_COMMANDS]:
            # We only save the config if the current stage is the fitting stage
            # (in this case we're training the network and we need the config)
            # or the user is e.g. testing the network and there was no previous
            # fitting stage. Otherwise, the code somehow crashes when run in slurm
            # and fitting and testing after another, due to some weird config
            # check failing.
            #super().setup(trainer, pl_module, stage)
            config_path = os.path.join(trainer.log_dir, self.config_filename)
            for logger in trainer.loggers:
                if logger.logger_name != 'mlflow':
                    logger.log_artifact(config_path)
