import os
import subprocess
from typing import Optional
import torch
import torchaudio
import numpy as np


def extract_audio_from_video_file(video_file_path, audio_file_path, start=0, length=-1):
    # The audio file extracted from the video file doesn't exist
    # and we have to create it first
    audio_file_dir = os.path.dirname(audio_file_path)
    os.makedirs(audio_file_dir, exist_ok=True)
    # Only extract if the audio file doesn't exist, yet
    convert_command = f'ffmpeg -y -hide_banner -loglevel error -i '\
        f'{video_file_path} -ss \'{start}ms\' -ar 16000 -ac 1 '
    if length != -1:
        convert_command += f'-t \'{length}ms\' '
    convert_command += audio_file_path
    subprocess.run(convert_command, shell=True)


def extract_spectrogram_from_audio_file(
    audio_file_path,
    spectrogram_file_path,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    normalize: bool = True
):
    waveform, sample_rate = torchaudio.load(audio_file_path)
    specgram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        n_fft=n_fft)
    # If we have two channels, we first compress the audio into mono.
    if len(waveform.shape) == 2:
        waveform = waveform.mean(0)
    specgram = specgram_transform(waveform)
    if normalize:
        spec_min = specgram.min()
        spec_max = specgram.max()
        specgram = (specgram - spec_min) / ((spec_max - spec_min) + 1e-8)
    np.save(spectrogram_file_path, specgram.cpu().numpy().squeeze())
