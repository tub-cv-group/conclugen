import numpy as np
import torch
from torchvision.transforms import ToPILImage


class ToPILImageVideo(object):

    def __init__(self):
        super().__init__()
        self._to_pil_image = ToPILImage()

    def __call__(self, x: np.array):
        x = np.transpose(x, (1, 0, 2, 3))
        result = [self._to_pil_image(elem) for elem in x]
        return result
