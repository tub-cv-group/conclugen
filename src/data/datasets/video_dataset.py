import os
from typing import Any, Dict, List
import time

import cv2
import numpy as np
import torch
import torchvision
from moviepy.editor import VideoFileClip
import cv2
from transformers import AutoTokenizer
from torchvision.utils import save_image
import time

from utils import video_util, constants as C
from data.datasets import AbstractDataset
from data.transforms import ContrastiveLearningViewGenerator, ToPILImageVideo, ToTensorVideo


class VideoBaseDataset(AbstractDataset):

    def __init__(
        self,
        paths_for_modality: Dict[str, str],
        modalities: List[str],
        split_modalities: List[str],
        multi_label: bool,
        transforms: Any,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.paths_for_modality = paths_for_modality
        self.spectrogram_to_tensor = torchvision.transforms.ToTensor()
        self.modalities = modalities
        self.split_modalities = split_modalities
        self.transforms = transforms
        self.multi_label = multi_label

    def __len__(self):
        return self.length

    def _put_data_into_batch(self, batch, data_key, data, filenames_key, filenames):
        if len(self.modalities) == 1:
            batch[C.BATCH_KEY_INPUTS] = data
            batch[C.BATCH_KEY_FILENAMES] = filenames
        else:
            # If we have multiple modalities we cannot just set them as the input directly,
            # instead we write each modality as a dict entry to the inputs
            batch[C.BATCH_KEY_INPUTS][data_key] = data
            batch[C.BATCH_KEY_FILENAMES][filenames_key] = filenames


class DynamicVideoClassificationDataset(VideoBaseDataset):

    def __init__(
        self,
        frames_2d_target_fps: int,
        frames_3d_target_fps: int,
        hugginface_tokenizer: str,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.frames_2d_target_fps = frames_2d_target_fps
        self.frames_3d_target_fps = frames_3d_target_fps
        self._to_pil_image_video = ToPILImageVideo()
        self._to_tensor_video = ToTensorVideo()
        if any([modality.startswith(C.BATCH_KEY_TEXTS) for modality in self.modalities]):
            self._tokenizer = AutoTokenizer.from_pretrained(hugginface_tokenizer)

    def _load_frames_2d(self, clip, fps, frame_indices: List[np.array], modality):
        # ###############################################################
        # ###############################################################
        # NOTE: The frames_2d get subsampled to 1 FPS most of the time!!!
        # NOTE: Don't print the frames here and wonder for the 1000th time
        # NOTE: why we have so few frames!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # ###############################################################
        # ###############################################################
        target_fps = min(fps, self.frames_2d_target_fps)
        subsampled_frames_indices = video_util.subsample_frames(frame_indices, fps, target_fps)
        frames_2d = [clip.get_frame(float(frame_idx) / fps) for frame_idx in subsampled_frames_indices]
        frames_2d = np.stack(frames_2d, axis=0)
        # We need to permute the dimensions to [3, num_frames, height, width] because that's what
        # subsequent code expects
        frames_2d = frames_2d.transpose(1, 0, 2, 3)
        frames_2d = self._to_pil_image_video(frames_2d)
        frames_2d = self._to_tensor_video(frames_2d)
        transforms = self.transforms.get(modality)
        if transforms is not None:
            # Apply temporally consistent transforms
            frames_2d = transforms(frames_2d)
        if isinstance(transforms, ContrastiveLearningViewGenerator):
            frames_2d = [frames.permute(1, 0, 2, 3) for frames in frames_2d]
        else:
            frames_2d = frames_2d.permute(1, 0, 2, 3)
        return frames_2d

    def _load_frames_3d(self, clip, fps, frame_indices: List[np.array], modality):
        target_fps = min(fps, self.frames_3d_target_fps)
        subsampled_frames_indices = video_util.subsample_frames(frame_indices, fps, target_fps)
        frames_3d = [clip.get_frame(float(frame_idx) / fps) for frame_idx in subsampled_frames_indices]
        # frames_3d is of shape [num_frames, 3, height, width]
        frames_3d = np.stack(frames_3d, axis=0)
        # We need to permute the dimensions to [3, num_frames, height, width] because that's what
        # subsequent code expects
        frames_3d = frames_3d.transpose(1, 0, 2, 3)
        frames_3d = self._to_pil_image_video(frames_3d)
        frames_3d = self._to_tensor_video(frames_3d)
        transforms = self.transforms.get(modality)
        if transforms is not None:
            # For frames_3d we apply temporally consistent transforms
            frames_3d = transforms(frames_3d)
        if isinstance(transforms, ContrastiveLearningViewGenerator):
            frames_3d = [frames.permute(1, 0, 2, 3) for frames in frames_3d]
        else:
            # For the sequence_collate_fn we need to permute the dimensions to be [num_frames, 3, height, width]
            frames_3d = frames_3d.permute(1, 0, 2, 3)
        return frames_3d

    def _load_spectrogram(self, key, modality):
        spectrogram_filename = key + '.npy'
        spectrogram_path = os.path.join(
            self.paths_for_modality[C.BATCH_KEY_AUDIO_SPECTROGRAMS], self.subset, spectrogram_filename)
        spectrogram = np.load(spectrogram_path)
        spectrogram = torch.from_numpy(spectrogram)
        # We need to add an additional batch dimension for transforms to work (they go for dim 2 as the time dim)
        spectrogram = spectrogram.unsqueeze(0)
        transforms = self.transforms.get(modality)
        if transforms is not None:
            spectrogram = transforms(spectrogram)
        # Remove the unecessary batch dimension
        spectrogram = spectrogram.squeeze()
        spectrogram = spectrogram.permute(1, 0)
        return spectrogram, spectrogram_filename

    def _load_text(self, key, modality):
        text_filename = key + '.txt'
        text_filepath = os.path.join(self.paths_for_modality[C.BATCH_KEY_TEXTS], self.subset, text_filename)
        with open(text_filepath, 'r') as text_file:
            # Only one line per text file
            text = text_file.readline()
        transforms = self.transforms.get(modality)
        if transforms is not None:
            before_text = text
            text = transforms(text)
            if isinstance(text, list):
                # Some augmentations return a list, but we always have only one sentence in there
                text = text[0]
        tokenized_text = self._tokenizer(text)
        return [torch.tensor(data) for data in tokenized_text.values()], text, text_filename

    def __getitem__(self, idx):
        # Puts in targets and annotations
        result = super().__getitem__(idx)
        key = self.sample_keys[idx]

        result.update({
            C.BATCH_KEY_INPUTS: {},
            C.BATCH_KEY_FILENAMES: {}
        })

        video_clip = None
        frame_indices = None

        # NOTE: Due to modality.startswith we can also use the keys with the augmentation suffixes since they
        # will be matched. This way, e.g. frames_2d_aug1 will be matched against frames_2d and the respective
        # transforms will be obtained in _load_frames_2d since transforms then is a dict like frames_2d_aug1: ...
        for modality in self.modalities:
            # Lazy load only once
            if frame_indices is None and\
                    (modality.startswith(C.BATCH_KEY_FRAMES_2D) or modality.startswith(C.BATCH_KEY_FRAMES_3D)):
                video_filename = key + '.mp4'
                video_path = os.path.join(self.paths_for_modality[C.BATCH_KEY_VIDEOS], self.subset, video_filename)
                video_clip = VideoFileClip(video_path)
                num_frames = int(video_clip.fps * video_clip.duration)
                # We employ this trick here to only load the frames that we really need,
                # thus we subsample the frame indices instead of loading the actual frames
                frame_indices = np.arange(num_frames)
            if modality.startswith(C.BATCH_KEY_FRAMES_2D_3D):
                frames_2d = self._load_frames_2d(video_clip, video_clip.fps, frame_indices, C.BATCH_KEY_FRAMES_2D)
                frames_3d = self._load_frames_3d(video_clip, video_clip.fps, frame_indices, C.BATCH_KEY_FRAMES_3D)
                self._put_data_into_batch(
                    result,
                    C.BATCH_KEY_FRAMES_2D_3D,
                    [frames_2d, frames_3d],
                    C.BATCH_KEY_FRAMES_2D_3D,
                    video_filename)
            if modality.startswith(C.BATCH_KEY_FRAMES_2D):
                frames_2d = self._load_frames_2d(video_clip, video_clip.fps, frame_indices, modality)
                self._put_data_into_batch(
                    result,
                    modality,
                    frames_2d,
                    modality,
                    video_filename)
            elif modality.startswith(C.BATCH_KEY_FRAMES_3D):
                frames_3d = self._load_frames_3d(video_clip, video_clip.fps, frame_indices, modality)
                self._put_data_into_batch(
                    result,
                    modality,
                    frames_3d,
                    modality,
                    video_filename)
            elif modality.startswith(C.BATCH_KEY_AUDIO_SPECTROGRAMS):
                spectrogram, spectrogram_filename = self._load_spectrogram(key, modality)
                self._put_data_into_batch(
                    result,
                    modality,
                    spectrogram,
                    modality,
                    spectrogram_filename)
            elif modality.startswith(C.BATCH_KEY_TEXTS):
                text_data, text, text_filename = self._load_text(key, modality)
                self._put_data_into_batch(
                    result,
                    modality,
                    text_data,
                    modality,
                    text_filename)

        if video_clip is not None:
            video_clip.close()

        result[C.BATCH_KEY_MODALITIES] = self.modalities

        return result


class PrecomputedFeaturesVideoClassificationDataset(VideoBaseDataset):

    def __init__(
        self,
        num_augmentated_samples_to_load: int=None,
        cache_features: str=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_augmentated_samples_to_load = num_augmentated_samples_to_load
        self._num_augmentations_for_modality = {}
        for modality in self.split_modalities:
            paths_for_modality = self.paths_for_modality[modality]
            if isinstance(paths_for_modality, List):
                self._num_augmentations_for_modality[modality] = len(paths_for_modality)
            else:
                self._num_augmentations_for_modality[modality] = 1
        assert cache_features in [None, 'cpu', 'gpu']
        self.cache_features = cache_features
        if self.cache_features is not None:
            # Set to False so we can load one sample from disk and then set the correct shape
            self.cache_features = False
            self.num_augmentated_samples_to_load = 1
            for modality in self.split_modalities:
                print(f'Loading precomputed features for modality {modality}...')
                num_augmentations = self._num_augmentations_for_modality[modality]
                sample = self._load_possibly_augmented_samples(0, self.sample_keys[0], modality)[0]
                cache = torch.zeros(
                    self.length,
                    num_augmentations,
                    *sample.shape)
                if cache_features == 'gpu':
                    cache = cache.cuda()
                print(f'Cache size for {modality}: {cache.shape}, {cache.element_size() * cache.nelement() / 1024**2} MB')
                cached_features_name = f'cached_features_{modality}'
                setattr(self, cached_features_name, cache)
                paths_for_modality = self.paths_for_modality[modality]
                for i in range(self.length):
                    key = self.sample_keys[i]
                    for j in range(num_augmentations):
                        if num_augmentations == 1:
                            path = paths_for_modality
                        else:
                            path =  paths_for_modality[j]
                        file_path = os.path.join(path, self.subset, key + '.npy')
                        features = torch.tensor(np.load(file_path))
                        if cache_features == 'gpu':
                            features = features.cuda()
                        cache[i][j] = features
            self.cache_features = cache_features
            self.num_augmentated_samples_to_load = num_augmentated_samples_to_load

    def _load_possibly_augmented_samples(self, sample_idx, key, modality):
        filename = key + '.npy'
        paths_for_modality = self.paths_for_modality[modality]
        # Only load augmented samples if we are in the training set. If the samples have augmentations
        # then the paths_for_modality will be a list (of directories containing the augmented samples).
        if isinstance(paths_for_modality, List) and self.subset == 'train':
            augmented_features_list = []
            num_augmentations = len(paths_for_modality)
            aug_indices = list(range(num_augmentations))
            for _ in range(self.num_augmentated_samples_to_load):
                aug_index = aug_indices.pop(np.random.randint(len(aug_indices)))
                samples_dir = os.path.join(paths_for_modality[aug_index], self.subset)
                features_path = os.path.join(samples_dir, filename)
                features = self._lookup_or_load_possibly_cached_features(modality, features_path, sample_idx, aug_index)
                augmented_features_list.append(features)
            final_features = augmented_features_list
            if len(final_features) == 1:
                # The case for non-SimCLR models
                final_features = final_features[0]
        elif isinstance(paths_for_modality, List):
            # Since the augmentations all have the same validaiton and test files, we just use the first
            features_path = os.path.join(paths_for_modality[0], self.subset, filename)
            final_features = self._lookup_or_load_possibly_cached_features(modality, features_path, sample_idx)
        else:
            features_path = os.path.join(paths_for_modality, self.subset, filename)
            final_features = self._lookup_or_load_possibly_cached_features(modality, features_path, sample_idx)
        return final_features, filename

    def _lookup_or_load_possibly_cached_features(self, modality, features_path, idx, aug_idx=0):
        # aug_idx 0 works also if there are no agumentations, because we have arrays of shape
        # (self.length, 1) in that case and (self.length, num_augmentations) otherwise.
        if not self.cache_features:
            return torch.tensor(np.load(features_path))
        cached_features_name = f'cached_features_{modality}'
        cache = getattr(self, cached_features_name)
        return cache[idx][aug_idx].clone().detach()

    def __getitem__(self, idx):
        # Puts in targets and annotations
        result = super().__getitem__(idx)
        key = self.sample_keys[idx]
        if len(self.modalities) > 1:
            result.update({
                C.BATCH_KEY_INPUTS: {},
                C.BATCH_KEY_FILENAMES: {}
            })

        frames_filename = key + '.npy'
        # NOTE: In the case that augmented precomputed features were requested, the modalities of this dataset will
        # be somehting like [frames_2d, frames_2d_aug1, frames_2d_aug2, ...]. But the dataset will only load samples
        # for the frames_2d key, since the others will fail the if-conditions below. The frames_2d key will then
        # load as many augmented samples as self.num_augmentations_to_load specifies. This is a bit weird but a quick
        # and easy solution so we will stick with it for now.
        for modality in self.modalities:
            # The frames_2d_3d key is not used by SimCLR since SimCLR either uses frames_2d or frames_3d. This means
            # that this part of the code does not need to implement the augmentation loading code.
            if modality == C.BATCH_KEY_FRAMES_2D_3D:
                video_features_2d, _ = self._load_possibly_augmented_samples(idx, key, C.BATCH_KEY_FRAMES_2D)
                video_features_3d, _ = self._load_possibly_augmented_samples(idx, key, C.BATCH_KEY_FRAMES_3D)
                self._put_data_into_batch(
                    result, C.BATCH_KEY_FRAMES_2D_3D,
                    [video_features_2d, video_features_3d],
                    C.BATCH_KEY_FRAMES_2D_3D,
                    frames_filename)
            if modality == C.BATCH_KEY_FRAMES_2D:
                video_features_2d, _ = self._load_possibly_augmented_samples(idx, key, C.BATCH_KEY_FRAMES_2D)
                self._put_data_into_batch(
                    result, C.BATCH_KEY_FRAMES_2D,
                    video_features_2d,
                    C.BATCH_KEY_FRAMES_2D,
                    frames_filename)
            elif modality == C.BATCH_KEY_FRAMES_3D:
                video_features_3d, _ = self._load_possibly_augmented_samples(idx, key, C.BATCH_KEY_FRAMES_3D)
                self._put_data_into_batch(
                    result, C.BATCH_KEY_FRAMES_3D,
                    video_features_3d,
                    C.BATCH_KEY_FRAMES_3D,
                    frames_filename)
            elif modality == C.BATCH_KEY_AUDIO_SPECTROGRAMS:
                spectrogram_features, spectrogram_filename = self._load_possibly_augmented_samples(
                        idx, key, C.BATCH_KEY_AUDIO_SPECTROGRAMS)
                self._put_data_into_batch(
                    result, C.BATCH_KEY_AUDIO_SPECTROGRAMS,
                    spectrogram_features,
                    C.BATCH_KEY_AUDIO_SPECTROGRAMS,
                    spectrogram_filename)
            elif modality == C.BATCH_KEY_TEXTS:
                text_features, text_filename = self._load_possibly_augmented_samples(
                    idx, key, C.BATCH_KEY_TEXTS)
                self._put_data_into_batch(
                    result, C.BATCH_KEY_TEXTS,
                    text_features,
                    C.BATCH_KEY_TEXTS,
                    text_filename)

            if modality == C.BATCH_KEY_FACIAL_LANDMARKS:
                raise NotImplementedError('Landmarks not implemented yet')
            if modality == C.BATCH_KEY_GLOVE_EMBEDDINGS:
                raise NotImplementedError('Glove embeddings not implemented yet')

        return result
