import logging

from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from models import ClassificationModel
from data.datamodules import ClassificationDataModule


_logger = logging.getLogger(__name__)


class SetClassWeightsOnModel(Callback):
    """This class takes care of taking the computed class weights from a classification
    datamodule and sets them on a classification model. This way, the model can
    take care of adjusting the loss function to counter imbalanced datasets.
    
    Make sure you are using a subclass of ClassificationDataModule and ClassificationModel.
    Then the only thing you need to do is to add this callback to the trainer's
    callbacks, e.g. through the `additional_callbacks` key in the root of the configs.
    
    NOTE: If this callback is present in the trainer's callbacks the CLI will
    take care of setting the loaded datamodule on the callback, that is not
    something you have to do.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._datamodule: ClassificationDataModule = None
        self._initialized = False

    @property
    def datamodule(self):
        return self._datamodule

    @datamodule.setter
    def datamodule(self, new_datamodule):
        if not isinstance(new_datamodule, ClassificationDataModule):
            raise Exception('Only `ClassificationDataModule`s allowed. Otherwise there might not be any class weights.')
        self._datamodule = new_datamodule

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        if self._initialized:
            return
        if self.datamodule is None:
            _logger.warning('The datamodule is `None`. Cannot retrieve weights to set on model.')
            return

        if not isinstance(pl_module, ClassificationModel):
            raise Exception(f'Expected a `ClassificationModel` not `{type(pl_module)}`.')
        if 'train' in self.datamodule.class_weights:
            pl_module.set_train_class_weights(self.datamodule.class_weights['train'])
        else:
            print('Train weights not found on model, skipping.')
        if 'val' in self.datamodule.class_weights:
            pl_module.set_val_class_weights(self.datamodule.class_weights['val'])
        elif 'validation' in self.datamodule.class_weights:
            pl_module.set_val_class_weights(self.datamodule.class_weights['validation'])
        else:
            print('Validation weights not found on model, skipping.')
        if 'test' in self.datamodule.class_weights:
            pl_module.set_test_class_weights(self.datamodule.class_weights['test'])
        else:
            print('Test weights not found on model, skipping.')
        if not 'train' in self.datamodule.class_weights\
           and not 'val' in self.datamodule.class_weights\
           and not 'validation' in self.datamodule.class_weights\
           and not 'test' in self.datamodule.class_weights:
            # If the user didn't provide any keywords we set the same weights
            # for all subsets
            pl_module.set_train_class_weights(self.datamodule.class_weights)
            pl_module.set_val_class_weights(self.datamodule.class_weights)
            pl_module.set_test_class_weights(self.datamodule.class_weights)
        self._initialized = True