import torch


class TimeApply(torch.nn.Module):

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, x: torch.Tensor):
        # Transform x from (channels, time, height, width) to (time, channels, height, width)
        result = x.permute((1, 0, 2, 3))
        for transform in self.transforms:
            result = torch.stack([transform(elem) for elem in result], dim=0)
        return result.permute((1, 0, 2, 3))
