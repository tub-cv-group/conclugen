from typing import Dict, Tuple, Union

from models.image_model import MEAN_STD_ARGS_TYPE, IMG_SIZE_ARG_TYPE
from . import AbstractDataModule


class ImagesDataModule(AbstractDataModule):
    """Base class for all datamodule working with images. It provides standard
    attributes necessary for most images, i.e. image size in [width, height],
    mean of the images per channel and standard deviation of the images per channel.
    """

    def __init__(
        self,
        img_size: IMG_SIZE_ARG_TYPE = (0, 0),
        mean: MEAN_STD_ARGS_TYPE = (0, 0, 0),
        std: MEAN_STD_ARGS_TYPE = (0, 0, 0),
        **kwargs
    ):
        """Init function of ImagesDataModule.

        Args:
            img_size (Union[Tuple[int, int], int], optional): The image size in 
                [width, height]. Defaults to (0, 0).
            mean (Tuple[float, float, float], optional): Mean of the images in
                the dataset (per channel). Can also be a dictionary if the model
                has multiple parts that require individual means.
                Defaults to (0.0, 0.0, 0.0).
            std (Tuple[float, float, float], optional): Standard deviation of
                the images in the dataset (per channel). Can also be a dictionary
                if the model has multiple parts that require individual stds.
                Defaults to (0.0, 0.0, 0.0).
        """
        # Need to be before call to __init__ so that super classe's init
        # finds these values (e.g. to init transforms)
        self.img_size = img_size
        if (isinstance(self.img_size, int) and self.img_size == 0) or\
            (isinstance(self.img_size, Tuple) and 0 in self.img_size):
            print(f'Image size {self.img_size} is either 0 or contains zeros - '
                'is this intended or a misconfiguration?')
        self.mean = mean
        self.std = std
        super().__init__(**kwargs)
