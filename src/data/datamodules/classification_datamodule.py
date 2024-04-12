from typing import List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler, SequentialSampler, WeightedRandomSampler

from data.datamodules import AbstractDataModule
from utils import constants as C


class ClassificationDataModule(AbstractDataModule):

    def __init__(
        self,
        num_classes: int,
        multi_label: bool,
        labels: List[str] = C.CAER_EXPRESSION_LABELS,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.labels = labels
        # Can be set later by the subclasses
        self._class_weights = {}
        self.compute_class_weights = False

    @property
    def class_weights(self) -> Dict[str, List[float]]:
        return self._class_weights

    @class_weights.setter
    def class_weights(self, new_weights: Dict[str, List[float]]):
        if 'train' in new_weights or 'val' in new_weights or 'test' in new_weights:
            for subset in new_weights.keys():
                assert len(new_weights[subset]) == self.num_classes,\
                    f'Lenght of class weights for `{subset}` has to match number of classes.'
        self._class_weights = new_weights
