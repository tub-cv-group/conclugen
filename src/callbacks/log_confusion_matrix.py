import os
from typing import Any

import torch

from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl

from utils import constants as C


class LogConfusionMatrix(Callback):

    def __init__(self, subsets_to_log=['val', 'test']):
        self.outputs = []
        self.subsets_to_log = subsets_to_log

    def _log_confusion_matrix(self, model, trainer, preds, targets, name, file_prefix):
        for logger in trainer.loggers:
            logger.log_confusion_matrix(
                preds,
                targets,
                name,
                # + 1 since this gets called before epoch is
                # incremented by the trainer
                trainer.current_epoch + 1,
                model.labels,
                file_prefix)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        unused: int = 0
    ) -> None:
        if 'train' in self.subsets_to_log:
            self.outputs.append(outputs)

    def on_train_epoch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule"
    ) -> None:
        self.outputs = []

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if 'train' not in self.subsets_to_log:
            return
        preds = torch.cat([pl_module.get_predicted_classes_from_outputs(preds[C.KEY_MODEL_OUTPUTS]) for preds in self.outputs])
        targets = torch.cat([tmp[C.KEY_TARGETS] for tmp in self.outputs])
        self._log_confusion_matrix(
            pl_module,
            trainer,
            preds.detach().cpu().numpy().tolist(),
            targets.detach().cpu().numpy().tolist(),
            'Train Confusion Matrix',
            'train')

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        unused: int = 0
    ) -> None:
        if 'val' in self.subsets_to_log:
            self.outputs.append(outputs)

    def on_validation_epoch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule"
    ) -> None:
        self.outputs = []

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if 'val' not in self.subsets_to_log:
            return
        preds = torch.cat([pl_module.get_predicted_classes_from_outputs(preds[C.KEY_MODEL_OUTPUTS]) for preds in self.outputs])
        targets = torch.cat([tmp[C.KEY_TARGETS] for tmp in self.outputs])
        self._log_confusion_matrix(
            pl_module,
            trainer,
            preds.detach().cpu().numpy().tolist(),
            targets.detach().cpu().numpy().tolist(),
            'Validation Confusion Matrix',
            'val')

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        unused: int = 0
    ) -> None:
        if 'val' in self.subsets_to_log:
            self.outputs.append(outputs)

    def on_test_epoch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule"
    ) -> None:
        self.outputs = []

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if 'test' not in self.subsets_to_log:
            return
        preds = torch.cat([pl_module.get_predicted_classes_from_outputs(preds[C.KEY_MODEL_OUTPUTS]) for preds in self.outputs])
        targets = torch.cat([tmp[C.KEY_TARGETS] for tmp in self.outputs])
        self._log_confusion_matrix(
            pl_module,
            trainer,
            preds.detach().cpu().numpy().tolist(),
            targets.detach().cpu().numpy().tolist(),
            'Test Confusion Matrix',
            'test')