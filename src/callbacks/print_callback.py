from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning import Callback


class PrintCallback(Callback):

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        print()
        print('==================== TRAINING ====================')
        print()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        print()
        print('================ FINISHED TRAINING ================')
        print()

    def on_val_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        print()
        print('=================== VALIDATING ===================')
        print()

    def on_val_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        print()
        print('=============== FINISHED VALIDATING ===============')
        print()

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        print()
        print('===================== TESTING =====================')
        print()

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        print()
        print('================= FINISHED TESTING =================')
        print()