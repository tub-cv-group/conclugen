import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Subset

from data.datamodules import AbstractDataModule


class KFoldDataModule(AbstractDataModule):

    def __init__(self, num_folds: int = None, random_state: int = 42,
        shuffle: bool = True, current_fold: int = 0, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_folds = num_folds
        if num_folds is not None:
            # Sometimes it's desired to not pass any folds
            self._skf = StratifiedKFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state)
        self.current_fold = current_fold
        self.random_state = random_state

