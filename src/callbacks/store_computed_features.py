from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule, Trainer

from utils import constants as C


class StoreComputedFeatures(Callback):

    def __init__(self, out_dir: str):
        super().__init__()
        self.out_dir = out_dir

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int
    ) -> None:
        features = outputs[C.KEY_RESULTS_FEATURES]