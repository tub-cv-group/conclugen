from typing import Union
from copy import deepcopy
import torchvision.models.resnet
import torchvision.models.vgg
import torchvision.models.video.resnet
import torch.nn as nn
import torch
import transformers
from pytorch_lightning.cli import instantiate_class
import pretorched

from utils import constants as C
from models.backbones import senet
from models.backbones import vgg_face as vgg_face_module
from models.backbones import tcn as tcn_module
from models.backbones import dav_enet as dav_enet_module
from models.backbones import TextEncoderHuggingface


def tcn(**kwargs):
    model = tcn_module.TCN(**kwargs)
    return model


def davenet(**kwargs):
    model = dav_enet_module.ResDavenet(**kwargs)
    return model


def resnet_classification_layer(self):
    return self.fc


def resnet_set_classification_layer(self, classification_layer):
    self.fc = classification_layer


def resnet_state_dict(self, destination=None, prefix='', keep_vars=False):
    result = self.super().state_dict(destination, prefix, keep_vars)


def resnet_forward(self, x):
    for layer in list(self.children())[:-1]:
        x = layer(x)
    features = x.view(x.shape[0], -1)
    logits = self.fc(features)
    results = {
        C.KEY_RESULTS_LOGITS: logits,
        C.KEY_RESULTS_FEATURES: features
    }
    return results


torchvision.models.resnet.ResNet.classification_layer = resnet_classification_layer
torchvision.models.resnet.ResNet.set_classification_layer = resnet_set_classification_layer
torchvision.models.resnet.ResNet.resnet_state_dict = resnet_state_dict
torchvision.models.resnet.ResNet.forward = resnet_forward


def resnet18(**kwargs):
    resnet = torchvision.models.resnet.resnet18(**kwargs)
    return resnet


def resnet34(**kwargs):
    resnet = torchvision.models.resnet.resnet34(**kwargs)
    return resnet


def resnet50(**kwargs):
    resnet = torchvision.models.resnet.resnet50(**kwargs)
    return resnet


def resnet101(**kwargs):
    resnet = torchvision.models.resnet.resnet101(**kwargs)
    return resnet


def resnet152(**kwargs):
    resnet = torchvision.models.resnet.resnet152(**kwargs)
    return resnet


torchvision.models.video.resnet.VideoResNet.classification_layer = resnet_classification_layer
torchvision.models.video.resnet.VideoResNet.set_classification_layer = resnet_set_classification_layer
torchvision.models.video.resnet.VideoResNet.resnet_state_dict = resnet_state_dict
torchvision.models.video.resnet.VideoResNet.forward = resnet_forward


def r2plus1d18(**kwargs):
    return torchvision.models.video.r2plus1d_18(**kwargs)


pretorched.models.resnext3D.ResNeXt3D.classification_layer = resnet_classification_layer
pretorched.models.resnext3D.ResNeXt3D.set_classification_layer = resnet_set_classification_layer
pretorched.models.resnext3D.ResNeXt3D.resnet_state_dict = resnet_state_dict
pretorched.models.resnext3D.ResNeXt3D.forward = resnet_forward

pretorched.models.resnet3D.ResNet3D.classification_layer = resnet_classification_layer
pretorched.models.resnet3D.ResNet3D.set_classification_layer = resnet_set_classification_layer
pretorched.models.resnet3D.ResNet3D.resnet_state_dict = resnet_state_dict
pretorched.models.resnet3D.ResNet3D.forward = resnet_forward


def resnext3d101(**kwargs):
    resnet = pretorched.resnext3d101(**kwargs)
    return resnet


def resnet3d101(**kwargs):
    # Loading the model like this modifies the forward function and deletes the
    # fc layer (renames it to last_linear), we reverse this here because we
    # need this different structure
    resnet = pretorched.resnet3d101(**kwargs)
    setattr(resnet.__class__, 'forward', resnet_forward)
    setattr(resnet.__class__, 'fc', resnet.last_linear)
    return resnet


def vgg_classification_layer(self):
    # Return the last layer of the classifier
    return self.classifier[6]


def vgg_set_classification_layer(self, classification_layer):
    self.classifier = nn.Sequential(*list(self.classifier.children())[:-1], classification_layer)


def vgg_forward(self,x):
    for layer in list(self.children())[:-1]:
        x = layer(x)
    features = x.view(x.shape[0], -1)
    logits = self.classifier(features)
    results = {
        C.KEY_RESULTS_LOGITS: logits,
        C.KEY_RESULTS_FEATURES: features
    }
    return results


torchvision.models.vgg.VGG.classification_layer = vgg_classification_layer
torchvision.models.vgg.VGG.set_classification_layer = vgg_set_classification_layer
torchvision.models.vgg.VGG.forward = vgg_forward


def vgg16(**kwargs):
    vgg16 = torchvision.models.vgg.vgg16(**kwargs)
    return vgg16


def vgg16_face(**kwargs):
    model = vgg_face_module.VGGFace(**kwargs)
    return model


def se_resnet_classification_layer(self):
    return self.classification_linear


def se_resnet_set_classification_layer(self, classification_layer):
    self.classification_linear = classification_layer


senet.SENet.classification_layer = se_resnet_classification_layer
senet.SENet.set_classification_layer = se_resnet_set_classification_layer


def se_resnet_50(**kwargs):
    net = senet.se_resnet50(**kwargs)
    return net


def se_resnet_101(**kwargs):
    net = senet.se_resnet101(**kwargs)
    return net


def se_resnet_152(**kwargs):
    net = senet.se_resnet152(**kwargs)
    return net


def se_resnext50_32x4d(**kwargs):
    net = senet.se_resnext50_32x4d(**kwargs)
    return net


def se_resnext101_32x4d(**kwargs):
    net = senet.se_resnext101_32x4d(**kwargs)
    return net


# Language transformers

def distillbert_emotion():
    pipeline = transformers.pipeline(
        'text-classification',
        'j-hartmann/emotion-english-distilroberta-base',
        device=0)
    return pipeline


def huggingface(model):
    return TextEncoderHuggingface(model)


def load_backbone_by_identifier(identifier: str, **kwargs):
    """Loads a given backbone by its identifier. It accepts an arbitrary number
    of parameters that will be passed to the network constructor. If there is
    no pretrained key in `kwargs` but the network supports this argument (like
    ResNet and VGG), it will be added automatically and set to True. If you don't
    want this, put `'pretrained': False` in `kwargs`.

    Args:
        identifier (str): the identifier of the network

    Returns:
        nn.Module: The instantiated model
    """
    constructor_func = globals().get(identifier)
    assert constructor_func, f'The specified backbone {identifier} is not known.'
    return constructor_func(**kwargs)


def load_backbone(identifier_or_dict: Union[str, dict]) -> nn.Module:
    """Loads the given backbone. You have several options how to load a backbone:

    * Plain string: The function will use the string to identify the network. If
      the network constructor allows to specify pretrained (like VGG and ResNet),
      it will be set to true.
    * Dictionary containing a `class_path` entry and `init_args` (as you're used
      to from jsongargparse): Pytorch Lightning's instantiate_class function will
      be used to instantiate the network.
    * Dictionary containing a `name` entry to identify the network and an arbitrary
      number of arguments. These will be delivered to the respective constructor
      of the network. This means that you need to know what argument the __init__
      of your network expects.

    Args:
        identifier_or_dict (Union[str, dict]): a string of the network name (e.g.
        resnet18), a dictionary in jsonargparse-style with `class_path` and 
        `init_args` entry or a dictionary with a `name` entry to identify the
        network and an additional arbitrary number of arguments that will be
        passed to the resepctive network constructor.

    Raises:
        Exception: If the network identifier is not known

    Returns:
        nn.Module: The instantiated network
    """
    if isinstance(identifier_or_dict, str):
        return load_backbone_by_identifier(identifier_or_dict)
    elif isinstance(identifier_or_dict, dict):
        if 'class_path' in identifier_or_dict:
            return instantiate_class(identifier_or_dict)
        else:
            copied_dict = deepcopy(identifier_or_dict)
            net_name = copied_dict.pop('name')
            args = copied_dict
            return load_backbone_by_identifier(net_name, **args)
    else:
        raise Exception(f'Illegal type for backbone config: {identifier_or_dict}')
