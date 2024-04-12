import os
from datetime import datetime  
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import logging

_logger = logging.getLogger(__name__)


class ModelCheckpointWithArtifactLogging(ModelCheckpoint):
    """This callback extends the ModelCheckpoint class by functionality to log
    the checkpoint weights automatically using the existing loggers. This way,
    checkpoints don't get lost when training crashes for some reason before
    the checkpoints can be logged in the on_train_epoch_end function.
    """

    def __init__(
        self,
        overwrite_existing: bool=True,
        log_model_every_n_epochs: int=5,
        log_best_model: bool=True,
        force_save_on_train_end: bool=True,
        **kwargs
    ):
        """Init function of model checkpoint that also logs the checkpoints
        as artifacts.

        Args:
            overwrite_existing (bool, optional): Whether to overwrite the existing
                logged checkpoint. This results in all checkpoints that are logged
                as artifacts to be renamed to best.ckpt. Defaults to True.
            log_model_every_n_epochs (int, optional): If set to greater than 1, the
                callback will additionally log the model using the loggers every n epochs
                (in some cases where the model is uploaded to a server, this might
                increase code runtime). Defaults to 5.
            log_best_model (bool, optional): Whether to log the best model at 
                the end of the training. To avoid erorrs, the current date + time
                will be appended to the model name. Defaults to True.
            force_save_on_train_end (bool, optional): If set, even if the loggers are set to not save checkpoints
                (e.g. for hyperparameter opt where it would spam the system), we can manually force to save the last
                and the best checkpoint anyways.
        """
        super().__init__(**kwargs)
        
        assert log_model_every_n_epochs > 0, 'Please provide a positive number greater than 0.'
        
        self.overwrite_existing = overwrite_existing
        self.log_model_every_n_epochs = log_model_every_n_epochs
        self.log_best_model = log_best_model
        self.force_save_on_train_end = force_save_on_train_end
        self.enabled = True

    def _log_checkpoint_at_path(
        self,
        trainer: pl.Trainer,
        filepath: str,
        filename: str
    ):
        if not os.path.exists(filepath):
            _logger.warning(f'The checkpoint path that was requested to be logged does not exist: {filepath}')
            return
        if not os.path.isfile(filepath):
            _logger.warning(f'The checkpoint path that was requested to be logged is not a file: {filepath}')
            return
        trainer.save_checkpoint(filepath)
        # Now store all artifacts at the checkpoint location
        for logger in trainer.loggers:
            logger.log_model(
                file_path=filepath,
                file_name_to_log=filename)

    def _log_checkpoint(self, trainer: pl.Trainer):
        if not trainer.is_global_zero or not self.enabled:
            return
        # We only do additional logging here, the actual logging based on a
        # metric is done within the loggers which the superclass of this class
        # calls after saving a checkpoint
        monitor_candidates = self._monitor_candidates(trainer)
        filepath = self.format_checkpoint_name(monitor_candidates)
        filename = os.path.basename(filepath)
        self._log_checkpoint_at_path(trainer, filepath, filename)

    def _additional_log_checkpoint(self, trainer: pl.Trainer):
        # trainer.current_epoch > 1 since epoch=00 is always saved since it's
        # the last one also when starting the training
        if self.log_model_every_n_epochs > 0 and\
           trainer.current_epoch > 1 and\
            trainer.current_epoch % self.log_model_every_n_epochs != 0:
            return
        if not self.enabled:
            return
        self._log_checkpoint(trainer)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        return_dict = super().on_train_epoch_end(trainer, pl_module)
        self._additional_log_checkpoint(trainer)
        return return_dict

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    )-> None:
        return_dict = super().on_train_end(trainer, pl_module)
        if self.force_save_on_train_end:
            self._additional_log_checkpoint(trainer)
        if self.log_best_model:
            best_model_path = self.best_model_path
            timestamp = datetime.now()
            str_date_time = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
            filename = f'best_{str_date_time}.ckpt'
            self._log_checkpoint_at_path(trainer, best_model_path, filename)
        return return_dict