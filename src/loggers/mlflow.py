import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas
import pytorch_lightning.loggers as pl_logger
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
import tempfile

from utils import file_util, dict_util, file_util, constants as C
from callbacks import ModelCheckpointWithArtifactLogging
from visualization.confusion_matrix import plot_confusion_matrix


class MLFlowLogger(pl_logger.MLFlowLogger):

    def __init__(self, ckpt_dir_name, **kwargs):
        super().__init__(**kwargs)
        # To allow to manually disable the upload of model weights to save time when they are not important
        self.upload_model_weights_enabeld = True
        self.ckpt_dir_name = ckpt_dir_name
        # Can be set to provide a prefix for metrics to log
        self.prefix = None

    def prepare_for_fit(self, config):
        pass

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

    def log_image(self, image: np.ndarray, filename: str):
        # self.run_id comes from the init_args in cli.py
        self.experiment.log_image(
            run_id=self.run_id,
            image=image,
            artifact_file=os.path.join('imgs', filename))

    def log_code(self, source_dir: str):
        self.experiment.log_artifacts(
            run_id=self.run_id,
            local_dir=source_dir,
            artifact_path='src')

    def log_model(self, file_path: str, file_name_to_log: str):
        # Not needed, since we use the ModelCheckpointWithArtifactLogging callback for storing checkpoints
        pass

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpointWithArtifactLogging) -> None:
        # The model checkpoint already stores the checkpoint exactly where the
        # mlflow logger would store it, i.e. we omit storing it here
        pass

    def log_confusion_matrix(self, preds, targets, name, epoch, labels, file_prefix):
        label_indices = list(range(len(labels)))
        matrix = confusion_matrix(targets, preds, labels=label_indices)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_confusion_matrix(matrix, labels, labels, fmt='d', ax=ax)
        fig_name = os.path.join('confusion_matrices',
                                f'{file_prefix}_epoch_{epoch}.png')
        self.log_figure(
            figure_name=fig_name,
            figure=fig)
        plt.close(fig)

    def _adjust_filename(self, prefixes: List[str], filename: str, step: str) -> str:
        if step:
            if '.' in filename:
                filename = filename[:-4] + '_' + str(step) + filename[-4:]
            else:
                filename = filename + '_' + str(step) + '.png'
        if prefixes:
            # If prefixes are given (e.g. ['compare_context', 'train']) we simply
            # create them as the folder structure in the filename so that they
            # get created on disk
            filename = os.path.join(os.path.join(*prefixes), filename)
        return filename

    def log_image(
        self,
        image: np.ndarray,
        filename: str,
        prefixes: List[str] = None,
        step: int = None,
        **kwargs
    ):
        if prefixes is not None:
            prefixes = ['imgs'] + prefixes
        else:
            prefixes = ['imgs']
        filename = self._adjust_filename(prefixes, filename, step)
        self.experiment.log_image(
            run_id=self.run_id,
            image=image,
            artifact_file=filename,
            **kwargs)

    def log_embedding(self, embeddings, labels, title):
        pass

    def log_histogram_3d(self, values, name=None, epoch=None, step=None):
        pass

    def log_figure(
        self, figure_name: str, figure: plt.Figure,
        prefixes: List[str] = None, overwrite: bool = False, step: int = None
    ):
        figure_name = self._adjust_filename(prefixes, figure_name, step)
        self.experiment.log_figure(
            run_id=self.run_id,
            figure=figure,
            artifact_file=f'imgs/{figure_name}')

    def log_artifact(self, file_path):
        self.experiment.log_artifact(self.run_id, file_path)

    def log_hyperparams(self, params) -> None:
        return super().log_hyperparams(params)

    def log_table(self, filename: str, tabular_data: pandas.DataFrame):
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_file_path = os.path.join(tmpdirname, filename)
            with open(temp_file_path, 'w+') as f:
                f.write(tabular_data.to_html())
            self.log_artifact(temp_file_path)

    def log_html(self, filename: str, html):
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_file_path = os.path.join(tmpdirname, filename)
            with open(temp_file_path, 'w+') as f:
                f.write(html)
            self.log_artifact(temp_file_path)
