import os
import glob
import pathlib
from typing import BinaryIO, List, Optional, Union
import yaml
import numpy as np
import cv2

import torch
from torchvision.utils import make_grid
from PIL import Image
from facenet_pytorch import MTCNN

from utils.file_util import split_path
from utils.dict_util import nested_set
from natsort import natsorted


def getMSSISM(i1, i2):
    C1 = 6.5025
    C2 = 58.5225
    # INITS
    I1 = np.float32(i1) # cannot calculate on one byte large values
    I2 = np.float32(i2)
    I2_2 = I2 * I2 # I2^2
    I1_2 = I1 * I1 # I1^2
    I1_I2 = I1 * I2 # I1 * I2
    # END INITS
    # PRELIMINARY COMPUTING
    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv2.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2
    sigma2_2 = cv2.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2
    sigma12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2                    # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2                    # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    ssim_map = cv2.divide(t3, t1)    # ssim_map =  t3./t1;
    mssim = np.mean(cv2.mean(ssim_map)[:3])       # mssim = average of ssim map
    return mssim


def make_image_have_3_channels(img: np.array) -> np.array:
    """Checks if the passed image img only has two channels, i.e. the length
    of its shape is 2. If yes, it gets converted to a 3-channel image by
    repeating the values.

    Args:
        img (np.array): the image to make have 3 channels

    Returns:
        np.array: the image img with definitely 3 channels
    """
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate(
            [img, img, img], axis=2)
    elif len(img.shape) != 3:
        raise Exception('Unsupported shape of image.')
    return np.array(img)


def crop_images_to_faces(source_path: str,
                         target_path: str,
                         detector: MTCNN,
                         confidence: float=0.9,
                         min_face_size: int=30,
                         crop_margin: int=5,
                         square_crops: bool=True,
                         source_prefix: str=None,
                         target_prefix: str=None,
                         img_ext: str='png'):
    """Automatically crops all images found using glob and the given img_ext
    at path source_path using the detector. It stores the cropped face images
    at the target_path with the same folder structure as source_path and also
    stores the coordinates of each individual cropped image. It prepends the
    target_prefix to all output images. Removes the source_prefix from the source
    images, if provided and not None.
    
    Expects the following folder structure in source_path:
    source_path
        sub_folder1
            img1
            img2
            img3
        sub_folder2
            img1
            img2

    Args:
        source_path (str): the source path containing the images
        target_path (str): the target path where to store the results
        detector (MTCNN): the detector
        confidence (float): the minimum confidence required to say it's a face.
            Defaults to 0.9.
        min_face_size (int): the minimum area a face needs to have. Defaults to 50.
        crop_margin (int): the margin to add around each face. Defaults to 25.
        square_crops (bool): whether to crop out square regions. Defaults to 'True'.
        source_prefix (str): will be removed from the source image filenames, 
            if provided. Defaults to None.
        target_prefix (str): a prefix to prepend to the output filenames.
            Not added when None. Defaults to None.
        img_ext (str, optional): The image file extension. Defaults to 'png'.
    """
    # Apply padding in all directions of the box found by MTCNN
    image_file_paths = glob.glob(os.path.join(source_path, '**', f'*.{img_ext}'), recursive=True)
    image_file_paths = natsorted(image_file_paths)
    frame_coordinates = {}
    for image_file_path in image_file_paths:
        # glob.glob includes source_path which is not what we need to recreate
        # the folder structure in the target path --> we make the path relative
        rel_img_file_path = os.path.relpath(image_file_path, source_path)
        img_dir = os.path.dirname(rel_img_file_path)
        target_img_dir = os.path.join(target_path, img_dir)

        print(f'Cropping {image_file_path} - storing in {target_img_dir}', end='\r')
        img = cv2.imread(image_file_path)
        img_filename = os.path.basename(image_file_path)
        boxes, probs = detector.detect(img)
        if boxes is None:
            continue
        probability = probs[0].item()
        if probability < confidence:
            continue
        # Only now make target directory so that we don't create a directory
        # when we are not detecting any faces in the contained images
        os.makedirs(target_img_dir, exist_ok=True)
        # Largest box is returned first
        box = boxes[0]
        box = [int(b) for b in box]
        x1, y1, x2, y2 = box
        x1 = x1 - crop_margin
        y1 = y1 - crop_margin
        x2 = x2 + crop_margin
        y2 = y2 + crop_margin
        width = x2 - x1
        height = y2 - y1
        # Make crops square
        if square_crops and width > height:
            diff = width - height
            y1 -= int(diff / 2)
            y2 += int(diff / 2)
        elif square_crops and height > width:
            diff = height - width
            x1 -= int(diff / 2)
            x2 += int(diff / 2)
        y1 = max(0, y1)
        y2 = min(img.shape[0], y2)
        x1 = max(0, x1)
        x2 = min(img.shape[1], x2)
        if source_prefix:
            img_filename = img_filename.replace(source_prefix, '')
        if target_prefix:
            img_filename = f'{target_prefix}_{img_filename}'
        out_img_path = os.path.join(target_img_dir, img_filename)
        width = x2 - x1
        height = y2 - y1
        if x1 < x2 and y1 < y2 and width * height > min_face_size:
            succeeded = cv2.imwrite(out_img_path, img[y1:y2, x1:x2])
            if not succeeded:
                raise Exception(f'Error writing image {out_img_path}.')
            # [:-1] since the last entry is the old filename like original_0000.png
            path_components = split_path(rel_img_file_path)[:-1]
            out_img_filename = os.path.basename(out_img_path)
            path_components.append(out_img_filename)
            nested_set(
                frame_coordinates,
                path_components,
                {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'prob': probability
                })
        print('', end='\r')

    with open(os.path.join(target_path, 'face_coordinates.yaml'), 'w+') as f:
        yaml.safe_dump(frame_coordinates, f, sort_keys=False)

@torch.no_grad()
def tensor_to_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    format: Optional[str] = None,
    **kwargs,
) -> Image.Image:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return Image.fromarray(ndarr)