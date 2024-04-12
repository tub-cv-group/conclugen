import numpy as np
import torch
from torchvision.transforms import Resize


class ResizeVideo(object):

    def __init__(self, size):
        super().__init__()
        self._resize = Resize(size)

    def __call__(self, x: np.array):
        x = np.transpose(x, (1, 0, 2, 3))
        result = torch.stack([self._resize(elem) for elem in x], dim=0)
        return result.permute((1, 0, 2, 3))
