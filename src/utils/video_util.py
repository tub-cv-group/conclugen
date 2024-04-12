import os
import subprocess
from typing import List, Union

import numpy as np
import torch
import ffmpeg


def subsample_frames(
    frames: Union[List, np.array, torch.Tensor],
    source_fps: int,
    target_fps: int
) -> Union[List, np.array, torch.Tensor]:
    """
    Subsamples frames from a list given a source FPS and a target FPS.

    Args:
        frames (list, np.array or torch.Tensor): The frames.
        source_fps (int): The source frames per second.
        target_fps (int): The target frames per second.

    Returns:
        A list/np.array/torch.Tensor of subsampled frames.
    """

    if target_fps > source_fps:
        raise ValueError("Target FPS cannot be greater than source FPS")

    if isinstance(frames, list):
        subsampled_frames = []
        frame_idx = 0

        while frame_idx < len(frames):
            subsampled_frames.append(frames[int(frame_idx)])
            frame_idx += source_fps / target_fps
        result = subsampled_frames
    elif isinstance(frames, np.ndarray):
        # Get total number of frames
        num_frames = frames.shape[0]
        # Calculate the total number of subsampled frames
        num_subsampled_frames = int(num_frames * (target_fps / source_fps))
        num_subsampled_frames = max(num_subsampled_frames, 1)
        # Generate equally spaced indices for subsampling
        indices = np.linspace(
            0, num_frames - 1, num_subsampled_frames, dtype=int)
        # Subsample frames
        result = frames[indices]
    elif isinstance(frames, torch.Tensor):
        num_frames = frames.shape[0]
        num_subsampled_frames = int(num_frames * (target_fps / source_fps))
        num_subsampled_frames = max(num_subsampled_frames, 1)
        indices = torch.linspace(
            0, num_frames - 1, num_subsampled_frames, dtype=int)
        result = frames[indices]
    return result


def cut_resize_video(
    input_video_file_path,
    output_video_file_path,
    with_sound,
    size=None,
    start=0,
    length=None
):
    # The audio file extracted from the video file doesn't exist
    # and we have to create it first
    audio_file_dir = os.path.dirname(input_video_file_path)
    os.makedirs(audio_file_dir, exist_ok=True)
    start = int(start)
    # Only extract if the audio file doesn't exist, yet
    convert_command = f'ffmpeg -y -hide_banner -loglevel error -i '\
        f'{input_video_file_path} -ss \'{start}ms\' -qscale 0 -ar 16000 '
    if size is not None:
        if isinstance(size, list) or isinstance(size, tuple):
            convert_command += f'-vf scale={size[0]}:{size[1]} '
        elif isinstance(size, int):
            convert_command += f'-vf scale={size}:{size} '
        else:
            raise Exception('Aspect-true resizing not yet supported.')
    if length is not None:
        length = int(length)
        convert_command += f'-t \'{length}ms\' '
    if not with_sound:
        convert_command += '-an '
    convert_command += output_video_file_path
    try:
        subprocess.run(convert_command, shell=True)
        return True
    except Exception as e:
        print(f'Error while converting video: {e}')
        return False
