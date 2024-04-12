import torch
import torch.nn as nn
from torchvision import models
import torch
# Choose the `x3d_s` model


class VideoEncoder(nn.Module):
    def __init__(
        self,
        num_channels: int=3,
        model_name=None,
        pretrained=True):

        super(VideoEncoder, self).__init__()
        if model_name == 'x3d_s':
            self.model = torch.hub.load(
                'facebookresearch/pytorchvideo', model_name, pretrained=pretrained)
            self.model.blocks[5].activation = nn.Identity()  # .ModuleList
            self.model.blocks[5].proj = nn.Identity()
        elif model_name == 'r2plus1d_18':
            self.model = models.video.r2plus1d_18(pretrained=pretrained)
            # Set to identity because we only want the features from the
            # convolutional layers. The final model can always put some additional
            # linear layers on top
            self.model.fc = nn.Identity()
            if num_channels == 4:
                new_first_layer = nn.Conv3d(
                    in_channels=4,
                    out_channels=self.model.stem[0].out_channels,
                    kernel_size=self.model.stem[0].kernel_size,
                    stride=self.model.stem[0].stride,
                    padding=self.model.stem[0].padding,
                    bias=False)
                # copy pre-trained weights for first 3 channels
                new_first_layer.weight.data[:,0:3] = self.model.stem[0].weight.data
                self.model.stem[0] = new_first_layer
        else:
            raise Exception(f'Unkown model name \'{model_name}\'.')

    def forward(self, x):
        out = self.model(x)
        return out
