from typing import Union

import torchvision.transforms.functional as F
import numpy as np
import numbers
import PIL.Image
import torch


class AspectRatioPadding(object):
    """Adds padding to an image in the requested aspect ratio but will try to
    resize the image first until padding is necessary.
    """

    def _get_padding(image, aspect_ratio):
        w, h = image.size

        current_aspect_ratio = w / float(h)
        scale_w = scale_h = 0
        padding_h = padding_w = 0

        if aspect_ratio > current_aspect_ratio:
            # Width is too small
            scale_w = aspect_ratio / current_aspect_ratio
            new_w = w * scale_w
            padding_w = (new_w - w) / 2.
        elif aspect_ratio < current_aspect_ratio:
            # Height is too small
            scale_h = current_aspect_ratio / aspect_ratio
            new_h = h * scale_h
            padding_h = (new_h - h) / 2.

        l_pad = padding_w if padding_w % 1 == 0 else padding_w+0.5
        t_pad = padding_h if padding_h % 1 == 0 else padding_h+0.5
        r_pad = padding_w if padding_w % 1 == 0 else padding_w-0.5
        b_pad = padding_h if padding_h % 1 == 0 else padding_h-0.5
        padding = [int(l_pad), int(t_pad), int(r_pad), int(b_pad)]

        return padding

    def __init__(self,
                 aspect_ratio: float = 1.0,
                 fill: int = 0,
                 padding_mode: str = 'constant'):
        """Init function of aspect ratio padding.

        Args:
            aspect_ratio (int, optional): Aspect ratio, width / height. Defaults to 1.
            fill (int, optional): What to pad with. Defaults to 0.
            padding_mode (str, optional): Choos from 'constant', 'edge',
                'reflect', 'symmetric'. Defaults to 'constant'.
        """
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        assert type(aspect_ratio) == float
        assert aspect_ratio > 0

        self.fill = fill
        self.padding_mode = padding_mode
        self.padding = []
        self.aspect_ratio = aspect_ratio

    def __call__(self, img: Union[PIL.Image.Image, torch.Tensor]):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        self.padding = AspectRatioPadding._get_padding(img, self.aspect_ratio)
        return F.pad(img, self.padding, self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)
