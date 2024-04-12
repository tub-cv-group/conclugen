import copy
import os
import tempfile
from typing import Any, Callable, Dict, List, Union

from attr import has
import torch
from torch import pca_lowrank
from torch import Tensor
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import umap
import tsnecuda
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import constants as C


class LogFeaturesVisualization(Callback):

    def __init__(
        self,
        reduce_method: List[str]=['tsne'],
        visu_frequence: str='epoch',
        subsets: List[str]=['val', 'test'],
        results_type: str=C.KEY_RESULTS_FEATURES,
        label: str=C.BATCH_KEY_EMOTIONS,
        axis_labels: List[str]=['Dim1', 'Dim2'],
        color_palette: str='Set2',
        restrict_to_annotation: Union[Dict[str, List], Dict[str, str], Dict[str, torch.Tensor]]=None,
        annotation_transform: Dict[str, str]=None,
        tsne_args: Dict[str, Any]={'n_components': 2, 'n_iter': 5000},
        umap_args: Dict[str, Any]={'n_components': 2},
        pca_args: Dict[str, Any]={'q': 512},
        log_dynamic_features: bool=False,
        legend: Union[str, bool]='auto'
    ):
        self.subsets = subsets
        self.visualization_frequency = visu_frequence
        self.reduce_method = reduce_method
        self.pca_args = pca_args
        self.available_methodes = {
            'tsne': tsnecuda.TSNE(**tsne_args),
            'umap': umap.UMAP(**umap_args)}
        self.results_type = results_type
        self.label = label
        self.axis_labels = axis_labels
        self.legend = legend
        self.color_palette = color_palette
        self.restrict_to_annotation = restrict_to_annotation
        # copy the dict so that we can create a label from its values (they get evaluated further below which inhibits this)
        self._restrict_to_annotation = copy.copy(restrict_to_annotation)
        if self.restrict_to_annotation is not None:
            for key, value in self.restrict_to_annotation.items():
                if isinstance(value, str) and value.startswith('lambda'):
                    self.restrict_to_annotation[key] = eval(value)
        self.annotation_transform = annotation_transform
        if self.annotation_transform is not None:
            for key, value in self.annotation_transform.items():
                self.annotation_transform[key] = eval(value)
        self.log_dynamic_features = log_dynamic_features
        # To cache the batch outputs
        self._features = {
            'train': None,
            'val': None,
            'test': None,
            'predict': None
        }
        self._labels = {
            'train': None,
            'val': None,
            'test': None,
            'predict': None
        }
        self._annotations = {
            'train': None,
            'val': None,
            'test': None,
            'predict': None
        }

    def _log_features(
        self,
        trainer: pl.Trainer,
        stage: str,
        subset
    ):
        features = self._features[subset]
        labels = self._labels[subset]
        annotations = self._annotations[subset]

        if self.pca_args is not None:
            U, S, V = pca_lowrank(features, **self.pca_args)
            # Compute the low-rank approximation
            features = features @ V

        identifier = f'{stage}_{self.label}'

        if self.restrict_to_annotation is not None:
            restricted = '_'.join([f'{key}={value}' for key, value in self._restrict_to_annotation.items()])
        else:
            restricted = None

        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        for method in self.reduce_method:
            fig_name = f'{method}_{stage}_{self.label}'
            reduce = self.available_methodes[method]
            total_tsne_features = reduce.fit_transform(features)
            total_data = np.hstack((total_tsne_features, labels[:, np.newaxis]))
            df_data = pd.DataFrame(
                total_data, columns=[self.axis_labels[0], self.axis_labels[1], self.label.capitalize()])
            temp_file = tempfile.NamedTemporaryFile()
            temp_file.name = f'{fig_name}.csv'
            print(f'Logging features using {method} for {identifier}...')
            df_data.to_csv(temp_file.name)
            fig = plt.figure()
            sns.scatterplot(
                data=df_data,
                hue=self.label.capitalize(),
                x=self.axis_labels[0],
                y=self.axis_labels[1],
                palette=self.color_palette,
                legend=self.legend)
            plt.title(f'{stage} {method} Visualization')
            if restricted is not None:
                fig_name += f'_{restricted}'
            fig_name += '.svg'
            for logger in trainer.loggers:
                logger.log_figure(fig_name, fig)
                logger.log_artifact(temp_file.name)
            os.remove(temp_file.name)

        if self.log_dynamic_features:
            print(f'Logging dynamic features for {identifier} to loggers...')
            if restricted is not None:
                identifier += f'_{restricted}'

            subset_ann_keys = list(annotations.keys())
            subset_ann_values = list(annotations.values())
            if len(subset_ann_keys) == 1:
                # Only one annotation, e.g. 'emotions' with [0, 1, 5, 3, 2, 0, ...] as labels
                labels = subset_ann_values[0].tolist()
            else:
                labels = [list(annotations.keys())]
                labels.extend(zip(*subset_ann_values))

            for logger in trainer.loggers:
                logger.log_embedding(features.detach().cpu().tolist(), labels, title=identifier)
            print(f'Logging dynamic features for {identifier} done!')

    # Shared method to store the features on batch end
    def _on_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: dict,
            batch: Any,
            batch_idx: int,
            subset: str
    ) -> None:
        if subset not in self.subsets:
            return

        annotations = batch[C.BATCH_KEY_ANNOTATIONS]
        current_labels = annotations[self.label]
        features = outputs[C.KEY_MODEL_OUTPUTS][self.results_type]

        if isinstance(current_labels, torch.Tensor):
            indices = set(list(range(current_labels.shape[0])))
        elif isinstance(current_labels, list):
            indices = set(list(range(len(current_labels))))
        else:
            raise ValueError(f'Unknown type for labels: {type(current_labels)}')
        if self.restrict_to_annotation is not None:
            for key, value in self.restrict_to_annotation.items():
                batch_entries = annotations[key]
                if callable(value):
                    indices = indices.intersection(set(torch.where(value(batch_entries))[0].tolist()))
                elif isinstance(value, list):
                    # Checks each entry if it is in value (i.e. a list of values)
                    entries_in_list = torch.tensor([entry in value for entry in batch_entries])
                    indices = indices.intersection(set(torch.where(entries_in_list)[0].tolist()))
                else:
                    value = torch.tensor(value)
                    indices = indices.union(set(torch.where(batch_entries == value)[0].tolist()))

        indices = torch.tensor(list(indices))
        # This is always a tensor
        features = features[indices]
        current_labels = self._reduce_data_to_indices(current_labels, indices)

        if self.annotation_transform is not None:
            label_transform = self.annotation_transform.get(self.label)
            if label_transform is not None:
                current_labels = label_transform(current_labels)

        annotations = self._reduce_data_to_indices(annotations, indices)
        if self.annotation_transform is not None:
            for key, value in self.annotation_transform.items():
                if key in annotations:
                    annotations[key] = value(annotations[key])

        self._features[subset] = self._extend_cached_data(self._features[subset], features)
        self._labels[subset] = self._extend_cached_data(self._labels[subset], current_labels)
        self._annotations[subset] = self._extend_cached_data(self._annotations[subset], annotations)

    def _reduce_data_to_indices(self, data, indices):
        if isinstance(data, torch.Tensor):
            return data[indices].detach().cpu().numpy()
        if isinstance(data, np.ndarray):
            return data[indices]
        if isinstance(data, list):
            return [data[idx] for idx in indices]
        if isinstance(data, dict):
            return {key: self._reduce_data_to_indices(value, indices) for key, value in data.items()}
        raise ValueError(f'Unknown type for data: {type(data)}')

    def _extend_cached_data(self, cache, new_data):
        if cache is None:
            return new_data
        if isinstance(cache, torch.Tensor):
            return torch.cat([cache, new_data])
        if isinstance(cache, np.ndarray):
            return np.concatenate([cache, new_data])
        if isinstance(cache, list):
            cache.extend(new_data)
            return cache
        if isinstance(cache, dict):
            for key, value in new_data.items():
                cache[key] = self._extend_cached_data(cache[key], value)
            return cache
        raise ValueError(f'Unknown type for cache: {type(cache)}')

    def _on_epoch_end(self, trainer: pl.Trainer, subset: str) -> None:
        if subset not in self.subsets:
            return

        if self.visualization_frequency == 'epoch':
            # For test somehow epoch is already incremented
            offset = 1 if subset == 'train' or subset == 'val' else 0
            self._log_features(trainer, f'{subset}_epoch={trainer.current_epoch + offset}', subset)
            self._features[subset] = None
            self._labels[subset] = None
            self._annotations[subset] = None
        if self.visualization_frequency == 'end':
            # Nothing to do here, we will log the features at the end of the training and they will
            # just get stacked further
            pass

    def _on_end(self, trainer: pl.Trainer, subset: str) -> None:
        if subset not in self.subsets:
            return

        if self.visualization_frequency == 'end':
            self._log_features(trainer, f'{subset}_end', subset)
            self._features[subset] = None
            self._labels[subset] = None
            self._annotations[subset] = None

    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: dict,
            batch: Any,
            batch_idx: int,
            unused: int = 0
    ) -> None:
        self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, 'train')

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None: 
        self._on_epoch_end(trainer, 'train')

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._on_end(trainer, 'train')

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: dict,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if not trainer.sanity_checking:
            self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, 'val')

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.sanity_checking:
            self._on_epoch_end(trainer, 'val')

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.sanity_checking:
            self._on_end(trainer, 'val')

    def on_test_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: dict,
            batch: Any,
            batch_idx: int,
            unused: int = 0
    ) -> None:
        self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, 'test')

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._on_end(trainer, 'test')

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._on_epoch_end(trainer, 'test')

    def on_predict_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: dict,
            batch: Any,
            batch_idx: int,
            unused: int = 0
    ) -> None:
        self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, 'predict')

    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._on_end(trainer, 'predict')