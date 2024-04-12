from typing import Dict, Tuple, Union

from models import AbstractModel


IMG_SIZE_ARG_TYPE = Union[Tuple[int, int], Dict[str, Union[int, Tuple[int, int]]], int]
MEAN_STD_ARGS_TYPE = Union[Tuple[float, float, float], Dict[str, Tuple[float, float, float]]]


class ImageModel(AbstractModel):
    """Base class for network modules that work on images.
    Provides provides the size of the images, the mean and the std.
    """

    def __init__(
        self,
        img_size: IMG_SIZE_ARG_TYPE,
        mean: MEAN_STD_ARGS_TYPE,
        std: MEAN_STD_ARGS_TYPE,
        **kwargs
    ):
        self.img_size = img_size
        self.mean = mean
        self.std = std
        super().__init__(**kwargs)
