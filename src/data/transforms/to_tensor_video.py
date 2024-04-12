import numpy as np
import torch
from torchvision.transforms import ToTensor


class ToTensorVideo(object):

    def __init__(self):
        super().__init__()
        self._to_tensor = ToTensor()

    def __call__(self, x: np.array):
        result = torch.stack([self._to_tensor(elem) for elem in x], dim=0)
        return result.permute((1, 0, 2, 3))
