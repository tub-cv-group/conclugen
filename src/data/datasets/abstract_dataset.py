from typing import Any, Dict
from torch.utils.data import Dataset

from utils import constants as C


class AbstractDataset(Dataset):

    def __init__(self, data_dir: str, annotations: Dict, subset: str, target_annotation: str):
        super().__init__()
        self.annotations = annotations
        self.sample_keys = list(self.annotations.keys())
        self.length = len(list(self.annotations.keys()))
        self.data_dir = data_dir
        self.subset = subset
        self.target_annotation = target_annotation

    def __getitem__(self, index) -> Any:
        result = {}
        key = self.sample_keys[index]
        if self.target_annotation is not None:
            result[C.BATCH_KEY_TARGETS] = self.annotations[key][self.target_annotation]
        result[C.BATCH_KEY_ANNOTATIONS] = self.annotations[key]
        return result