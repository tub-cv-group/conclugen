import os
from typing import Any

from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl


class LogWeightsHistograms(Callback):

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.is_global_zero:
            return
        for name, layer in pl_module.named_modules():
            if "activ" in name:
                continue

            if not hasattr(layer, "weight") and not hasattr(layer, "bias"):
                continue

            wname = "%s.%s" % (name, "weight")
            bname = "%s.%s" % (name, "bias")

            for logger in trainer.loggers:
                if hasattr(layer, "weight") and layer.weight is not None:
                    logger.log_histogram_3d(layer.weight.detach().cpu().numpy(), name=wname, step=trainer.current_epoch)
                if hasattr(layer, "bias") and layer.bias is not None:
                    logger.log_histogram_3d(layer.bias.detach().cpu().numpy(), name=bname, step=trainer.current_epoch)