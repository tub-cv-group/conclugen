from pytorch_lightning import Callback
import pytorch_lightning as pl


# A callback class that uploads the final stored model weights to CometML

class UploadCheckpointsToCometOnFitEnd(Callback):

    def __init__(self, ckpt_dir):
        super().__init__()
        self.ckpt_dir = ckpt_dir

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # Lazy import to avoid cyclic dependencies
        from loggers import CometLogger
        for logger in trainer.loggers:
            if isinstance(logger, CometLogger):
                if not logger.upload_model_weights_enabeld:
                    print('Uploading model weights to CometML was disabled - enabling to upload final models.')
                logger.upload_model_weights_enabeld = True
                logger.log_model(file_path=self.ckpt_dir, file_name_to_log='model')
                break
