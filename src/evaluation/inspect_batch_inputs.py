import os
import shutil

import torch
from torchvision.utils import save_image, make_grid
from pytorch_lightning.core import LightningDataModule
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from PIL.Image import Image
import cv2

from data.transforms import NormalizeInverse
from utils.image_util import tensor_to_image
from utils.dict_util import flatten


def _get_imgs_from_tensor(tensor, normalize_inverse):
    imgs = []
    if len(tensor.shape) == 3 or len(tensor.shape) == 4:
        # Either tensor is [C, H, W] or [B, C, H, W]
        imgs.append(normalize_inverse(tensor))
    elif len(tensor.shape) == 5:
        # Here we also have the time domain, we need to check
        # which dimension is it, could be [B, T, C, H, W] or
        # [B, C, T, H, W]
        if tensor.shape[1] == 3:
            # if shape[1] is 3, i.e. the channels, then we need to loop over the
            # time in shape[2]
            for t in range(tensor.shape[2]):
                imgs.append(normalize_inverse(tensor[:, :, t]))
        elif tensor.shape[2] == 3:
            # vice versa
            for t in range(tensor.shape[1]):
                imgs.append(normalize_inverse(tensor[:, t]))
        else:
            raise Exception('Tensor shape has length 5 and we '\
                'expected dim 1 or 2 to be the channels, i.e. of size 3, '\
                'but neither matched. This seems to be an unimplemented case.')
    else:
        print(f'The shape of the input tensor is {len(tensor.shape)}, '\
            'which is currently unsupported. You need to implement it yourself.')
    return imgs


def _extract_batch_inputs(trainer: Trainer, model, dataloader, limit, subset):
    if type(limit) == float:
        num_batches = int(len(dataloader) * limit)
    else:
        num_batches = limit

    outdir = trainer.default_root_dir
    print(f'Extracting input images of {num_batches} batches to image directory '
          f'of {outdir}.')

    normalize_inverse = NormalizeInverse(model.mean, model.std)
    
    for i, batch in enumerate(dataloader):
        if i > num_batches:
            break
        inputs = model.extract_inputs_from_batch(batch, i)
        imgs = []
        if isinstance(inputs, dict):
            flattened_inputs = flatten(inputs)
            # Get list of values so the subsequent code still works
            inputs = list(flattened_inputs.values())
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            # Some datasets have multiple inputs
            for input in inputs:
                if isinstance(input, Image):
                    # Only do this for images
                    imgs.append(normalize_inverse(input))
                elif isinstance(input, torch.Tensor):
                    imgs.extend(_get_imgs_from_tensor(input, normalize_inverse))
        if isinstance(inputs, torch.Tensor):
            imgs = _get_imgs_from_tensor(inputs, normalize_inverse)
        if len(imgs) > 0:
            imgs = torch.cat(imgs)
            img = tensor_to_image(make_grid(imgs, nrow=model.batch_size))
            # In case that the batch consists of multiple images/inputs, we
            # pair the inputs together. Otherwise the first grid would be input
            # of type A, the next grid would be all images of type B, ...
            # imgs is [num_input_modalities, batch_size, C, H, W]
            # We go over batch size and pair the different inputs
            #for j in range(len(imgs[0][0])):
                # We are concatenating the inputs now so that
                # all the input images are next to each other
            #    save_image(make_grid([_imgs[j] for _imgs in imgs], len(imgs)),
            #        os.path.join(outdir, f'batch_{i}.jpg'))
        else:
            img = tensor_to_image(make_grid(inputs, 4))
        filename = f'batch_{i}.jpg'
        prefixes = ['batch_inputs', subset]
        for logger in trainer.loggers:
            logger.log_image(img, filename=filename, prefixes=prefixes)
    
    print('Finished.')


def inspect_batch_inputs(
        model: LightningModule,
        datamodule: LightningDataModule,
        trainer: Trainer,
        limit_batches: int = 10
):
    """Extracts the image inputs from the batches for inspection.
    Currently, only images as input are supported.

    Args:
        model (LightningModule): the model that provides the extract inputs function
        datamodule (LightningDataModule): the datamodule that provides the batches
        trainer (Trainer): not used currently
        limit_batches (int, optional): how many batches to process. Defaults to 10.
    """
    datamodule.prepare_data()
    datamodule.setup()

    dataloaders = [
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        datamodule.test_dataloader()]
    subsets = ['train', 'val', 'test']

    for subset, dataloader in zip(subsets, dataloaders):
        _extract_batch_inputs(trainer, model, dataloader, limit_batches, subset)

