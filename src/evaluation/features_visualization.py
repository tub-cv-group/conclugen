import os
import shutil

import torch
from pytorch_lightning.core import LightningDataModule
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer

from callbacks import LogFeaturesVisualization


def features_visualization(
    model: LightningModule,
    datamodule: LightningDataModule,
    trainer: Trainer,
    ckpt_path: str,
    reduce_methode: list
):
    '''
    visualizes the features of the model processed with TSNE
    '''
    datamodule.prepare_data()
    datamodule.setup()

    # Add logging of visual maps of features reduced with T-SNE
    additional_callback = LogFeaturesVisualization(reduce_methode=reduce_methode)
    trainer.callbacks.append(additional_callback)

    if ckpt_path is not None:
        print(f'Loading model from {ckpt_path}')
        ckpt = torch.load(ckpt_path)
        missing, unxepected = model.load_state_dict(ckpt['state_dict'], strict=False)
        if len(missing) > 0:
            print(f'Missing keys: {missing}')
        if len(unxepected) > 0:
            print(f'Unexpected keys: {unxepected}')

    additional_callback.test_data = True
    additional_callback.train_data = False
    additional_callback.val_data = False
    trainer.predict(model, datamodule.test_dataloader())
    additional_callback.test_data = False
    additional_callback.train_data = True
    additional_callback.val_data = False
    trainer.predict(model, datamodule.train_dataloader())
    additional_callback.test_data = False
    additional_callback.train_data = False
    additional_callback.val_data = True
    trainer.predict(model, datamodule.val_dataloader())