import numpy as np
from pytorchvideo.transforms import RandomResizedCrop


class RandomResizedCropVideo(RandomResizedCrop):

    def __init__(self, size, ratio, **kwargs):
        target_height = size[0]
        target_width = size[1]
        super().__init__(target_height=target_height, target_width=target_width, aspect_ratio=ratio, **kwargs)
