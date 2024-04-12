import torch
from torchvision.transforms import ColorJitter, ToPILImage, ToTensor


class VideoColorJitter(ColorJitter):

    def __init__(self, p: float=0.5, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def forward(self, x):
        x = x.permute((1, 0, 2, 3))
        if self.p < torch.rand(1):
            x =  super().forward(x)
        return x.permute((1, 0, 2, 3))
