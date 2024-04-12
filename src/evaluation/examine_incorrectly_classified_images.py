import os
import shutil

from pytorch_lightning.core import LightningDataModule
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer

from callbacks import LogIncorrectlyClassifiedImages


def examine_incorrectly_classified_images(
    model: LightningModule,
    datamodule: LightningDataModule,
    trainer: Trainer,
    ckpt_path: str
):
    datamodule.prepare_data()
    datamodule.setup()

    # Add logging of incorrectly classified images
    trainer.callbacks.append(LogIncorrectlyClassifiedImages())

    num_batches = trainer.limit_predict_batches

    print(f'Extracting attention maps for {num_batches} batches (due to trainer.limit_predict_batches).')

    # We simply run prediction (to avoid all other logging) on the test dataloader
    trainer.predict(model, datamodule.test_dataloader(), ckpt_path=ckpt_path)
