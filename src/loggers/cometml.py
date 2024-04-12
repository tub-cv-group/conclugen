import os
from typing import Dict, List, Optional, Union, Any
from matplotlib import pyplot as plt
import numpy as np
import pandas
import pytorch_lightning.loggers as pl_logger
from sklearn.metrics import confusion_matrix
import torch
from argparse import Namespace

from lightning_fabric.utilities.logger import _convert_params
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from callbacks import ModelCheckpointWithArtifactLogging
from utils import file_util, dict_util
from utils import constants as C


class CometLogger(pl_logger.CometLogger):

    def __init__(self, upload_model_weights_enabeld=True, **kwargs):
        super().__init__(**kwargs)
        # We have to sanitze Pytorch Lightning _kwargs due to a bug since
        # version 1.6 (see https://github.com/PyTorchLightning/pytorch-lightning/issues/12529)
        self._kwargs = {
            k: self._kwargs[k] for k in self._kwargs
            if k not in ['agg_key_funcs', 'agg_default_func']
        }
        # To allow to manually disable the upload of model weights to save time when they are not important
        self.upload_model_weights_enabeld = upload_model_weights_enabeld
        # Can be set to provide a prefix for metrics to log
        self.prefix = None

    def prepare_for_fit(self, config):
        config[C.KEY_COMETML_RUN_ID] = self.experiment.get_key()
        # To avoid these unnecessary long Comet ML experiment summaries
        # We cannot set this in the config probably
        #self.experiment.display_summary_level = 0
        # Write out experiment key so that training can be resumed within the
        # same experiment run of Comet ML
        cometml_experiment_key_file_path = file_util.get_mlflow_artifact_path(
            run_id=config[C.KEY_RUN_ID],
            artifact_name='cometml.txt'
        )
        with open(cometml_experiment_key_file_path, 'w+')\
                as cometml_experiment_key_file:
            data = self.experiment.get_key()
            cometml_experiment_key_file.write(data)

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        self.experiment.log_parameters(params)

    def log_metrics(
        self,
        metrics: Dict[str, Union[torch.Tensor, float]],
        step: Optional[int] = None
    ) -> None:
        if self.prefix is not None:
            metrics = {self.prefix + k: v for k, v in metrics.items()}
        # We need to flatten the dictionary because when using the ClasswiseWrapper
        # it will return nested metrics which are not supported for logging
        metrics = dict_util.flatten(metrics)
        return super().log_metrics(metrics, step)

    def log_code(self, source_dir: str):
        self.experiment.log_code(folder=source_dir)

    def log_model(self, file_path: str, file_name_to_log: str):
        if self.upload_model_weights_enabeld:
            print('Uploading model files to CometML.')
            self.experiment.log_model(
                name='model',
                file_or_folder=file_path,
                file_name=file_name_to_log,
                overwrite=True,
                prepend_folder_name=False)
        else:
            print('Skipping upload of model weights to CometML (upload_model_weights_enabeld is False on logger).')

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpointWithArtifactLogging) -> None:
        path = checkpoint_callback.best_model_path
        filename = os.path.basename(path)
        self.log_model(file_path=path, file_name_to_log=filename)

    def log_confusion_matrix(self, preds, targets, name, epoch, labels, file_prefix):
        computed_confusion_matrix = confusion_matrix(
            y_true=targets,
            y_pred=preds,
            labels=list(range(len(labels))))
        self.experiment.log_confusion_matrix(
            matrix=computed_confusion_matrix,
            title=f'{name} Epoch {epoch}',
            file_name=f'{file_prefix}_epoch_{epoch}.json',
            labels=labels)

    def log_image(
        self,
        image: np.ndarray,
        filename: str,
        prefixes: List[str]=None,
        step: int=None,
        **kwargs
    ):
        if prefixes is not None:
            for i in range(len(prefixes) - 1, 0, -1):
                filename = prefixes[i] + '_' + filename
        self.experiment.log_image(
            image_data=image,
            name=filename,
            step=step,
            **kwargs)

    def log_histogram_3d(self, values, name=None, epoch=None, step=None):
        self.experiment.log_histogram_3d(values, name, epoch, step)

    def log_figure(
        self,
        figure_name: str,
        figure: plt.Figure,
        prefixes: List[str]=None,
        overwrite: bool=False,
        step: int=None,
        **kwargs
    ):
        if prefixes:
            str_prefix = '_'.join(prefixes)
            figure_name = str_prefix + '_' + figure_name
        self.experiment.log_figure(
            figure_name=figure_name,
            figure=figure,
            overwrite=overwrite,
            step=step,
            **kwargs)

    def log_embedding(self, embeddings, labels, title):
        self.experiment.log_embedding(
            vectors=embeddings,
            labels=labels,
            title=title)

    def log_artifact(self, file_path, step=None):
        # log_artifact method of Comet-ML wants the actual object, and
        # log_asset can be used to provide the path to the file
        self.experiment.log_asset(file_path, step=step)

    def log_table(self, filename: str, tabular_data: pandas.DataFrame):
        self.experiment.log_table(filename, tabular_data)

    def log_html(self, filename: str, html_data):
        self.experiment.log_html(filename, html_data)

    def add_tag(self, tag):
        self.experiment.add_tag(tag)