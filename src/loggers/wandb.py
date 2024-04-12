from fileinput import filename
import os
from typing import List
from matplotlib import pyplot as plt
import numpy as np
import pytorch_lightning.loggers as pl_logger
import wandb

from callbacks import ModelCheckpointWithArtifactLogging
from utils import file_util
from utils import constants as C


class WandbLogger(pl_logger.WandbLogger):

    def __init__(self, **kwargs):
        os.environ['WANDB_SILENT'] = 'true'
        super().__init__(**kwargs)

    def prepare_for_fit(self, config):
        config[C.KEY_WANDB_RUN_ID] = self.experiment.id
        # Write out experiment key so that training can be resumed within the
        # same experiment run of Comet ML
        experiment_key_file_path = file_util.get_mlflow_artifact_path(
            run_id=config[C.KEY_RUN_ID],
            artifact_name='wandb.txt'
        )
        with open(experiment_key_file_path, 'w+')\
                as experiment_key_file:
            data = self.experiment.id
            experiment_key_file.write(data)

    def log_code(self, source_dir: str):
        self.experiment.log_code(root=source_dir)

    def log_model(self, file_path: str, file_name_to_log: str):
        # The WandB logger already has an implementation for after_save_checkpoint
        pass

    def log_confusion_matrix(self, preds, targets, name, epoch, labels, file_prefix):
        self.experiment.log(
            {name : wandb.plot.confusion_matrix(
                y_true=targets,
                preds=preds,
                class_names=labels)
            },
            step=epoch
        )

    def log_image(
        self,
        image: np.ndarray,
        filename: str,
        prefixes: List[str]=None,
        step: int=None,
        **kwargs
    ):
        if prefixes is not None:
            for i in range(len(prefixes), 0, -1):
                filename = prefixes[i] + '_' + filename
        image = wandb.Image(image, **kwargs)
        self.experiment.log({filename: image}, step=step, **kwargs)

    def log_histogram_3d(self, values, name=None, step=None):
        histogram = wandb.Histogram(values)
        self.experiment.log({name: histogram})

    def log_figure(
        self,
        figure_name: str,
        figure: plt.Figure,
        prefixes: List[str]=None,
        overwrite: bool=False,
        step: int=None,
        **kwargs,
    ):
        if prefixes:
            str_prefix = '_'.join(prefixes)
            figure_name = str_prefix + '_' + figure_name
        self.experiment.log(
            {figure_name: figure},
            step=step,
            **kwargs)
    
    def log_artifact(self, file_path, step=None):
        # log_artifact method of Comet-ML wants the actual object, and
        # log_asset can be used to provide the path to the file
        self.experiment.save(file_path)