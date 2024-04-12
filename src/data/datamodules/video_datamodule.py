from copy import copy, deepcopy
import os
from typing import Dict, List
import math
import time
from copy import deepcopy
from abc import abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
import yaml

from data.samplers import RangeSampler
from data.datamodules import DataConfig
from data.datamodules import ImagesDataModule, ClassificationDataModule
from data.datasets import DynamicVideoClassificationDataset, PrecomputedFeaturesVideoClassificationDataset
from data.transforms import NoOp
from data.collators import sequence_collate_fn
from utils import file_util, audio_util, features_util, input_util, constants as C
from utils.instantiation_util import instantiate_transforms_tree as ite
from models.backbones import backbone_loader


class VideoBaseDataModule(ImagesDataModule):

    def __init__(
        self,
        modalities: List[str],
        num_mels: int = None,
        resize_scale: int = None,
        inference_start_pos: float = None,
        subsample_audio: bool = False,
        max_embedding_length: int = None,
        feature_precomputation_config: Dict = None,
        num_augmented_samples_to_load: int = None,
        crop_face: bool = False,
        preprocessing_batch_size: int = None,
        cache_features: str = None,
        **kwargs
    ):
        """Init function of VideoBaseDataModule.
        This class provides some common attributes and methods for datamodules
        that work on and provide videos of data.

        Args:
            modalities (str): The modalities to use. Check the inheriting child
                datamodule to see the available modalities.
            inference_start_pos (float): Modifier to alter the starting position
                in the video video during inference. This only applies when
                not the whole video is processed during inference but only one
                chunk. This modifier is then multiplied onto the video length
                and this way not always the first chunk of the video is selected.
            subsample_audio (bool): Whether to subsample the audio in the same
                manner as the video frames. Defaults to False.
        """
        super().__init__(**kwargs)

        self._data_configs = {
            C.DATA_KEY_ANNOTATIONS: DataConfig(
                dependencies=[],
                data_paths=[C.FILENAME_ANNOTATIONS],
                data_counts=None,
                extensions=['yaml'],
                precomputation_batch_keys=None,
                precomputation_backbone_keys=None,
                precomputation_funcs=None,
                precomputation_transforms=None
            ),
            C.DATA_KEY_VIDEOS_ORIGINAL: DataConfig(
                dependencies=[],
                data_paths=[C.DIRNAME_VIDEOS_ORIGINAL],
                data_counts=None,
                extensions=['mp4'],
                precomputation_batch_keys=None,
                precomputation_backbone_keys=None,
                precomputation_funcs=None,
                precomputation_transforms=None
            ),
            C.DATA_KEY_VIDEOS: DataConfig(
                dependencies=[C.DATA_KEY_VIDEOS_ORIGINAL],
                data_paths=[C.DIRNAME_VIDEOS],
                data_counts=None,
                extensions=['mp4'],
                precomputation_batch_keys=None,
                precomputation_backbone_keys=None,
                precomputation_funcs=None,
                precomputation_transforms=None
            ),
            C.DATA_KEY_FRAMES_2D_3D: DataConfig(
                dependencies=[C.DATA_KEY_FRAMES_2D, C.DATA_KEY_FRAMES_3D],
                # Empty because frames_2d and frames_3d will be checked individually
                data_paths=None,
                data_counts=None,
                extensions=None,
                # None because this modality will be split into frames_2d and frames_3d
                precomputation_batch_keys=None,
                precomputation_backbone_keys=None,
                precomputation_funcs=None,
                precomputation_transforms=None
            ),
            C.DATA_KEY_FRAMES_2D: DataConfig(
                dependencies=[C.DATA_KEY_VIDEOS],
                data_paths=[C.DIRNAME_FRAMES_2D],
                data_counts=None,
                extensions=['npy'],
                precomputation_batch_keys=C.BATCH_KEY_FRAMES_2D,
                precomputation_backbone_keys=C.BACKBONE_KEY_FRAMES_2D,
                precomputation_funcs=self._process_video_2d_batch
            ),
            C.DATA_KEY_FRAMES_3D: DataConfig(
                dependencies=[C.DATA_KEY_VIDEOS],
                data_paths=[C.DIRNAME_FRAMES_3D],
                data_counts=None,
                extensions=['npy'],
                precomputation_batch_keys=C.BATCH_KEY_FRAMES_3D,
                precomputation_backbone_keys=C.BACKBONE_KEY_FRAMES_3D,
                precomputation_funcs=self._process_video_3d_batch
            ),
            C.DATA_KEY_AUDIO: DataConfig(
                dependencies=[C.DATA_KEY_VIDEOS_ORIGINAL],
                data_paths=[C.DIRNAME_AUDIO],
                data_counts=None,
                extensions=['wav'],
                precomputation_batch_keys=None,
                precomputation_backbone_keys=None,
                precomputation_funcs=None,
                precomputation_transforms=None
            ),
            C.DATA_KEY_AUDIO_SPECTROGRAMS: DataConfig(
                dependencies=[C.DATA_KEY_AUDIO],
                data_paths=[C.DIRNAME_AUDIO_SPECTROGRAMS],
                data_counts=None,
                extensions=['npy'],
                precomputation_batch_keys=C.BATCH_KEY_AUDIO_SPECTROGRAMS,
                precomputation_backbone_keys=C.BACKBONE_KEY_AUDIO_SPECTROGRAMS,
                precomputation_funcs=self._process_specrograms_batch
            ),
            C.DATA_KEY_FACIAL_LANDMARKS: DataConfig(
                dependencies=[C.DATA_KEY_VIDEOS_ORIGINAL],
                data_paths=[C.DIRNAME_FACIAL_LANDMARKS],
                data_counts=None,
                extensions=['npy'],
                precomputation_batch_keys=C.BATCH_KEY_FACIAL_LANDMARKS,
                precomputation_backbone_keys=C.BACKBONE_KEY_FACIAL_LANDMARKS,
                # Precomputation not yet supported
                precomputation_funcs=None,
                precomputation_transforms=None
            ),
            C.DATA_KEY_GLOVE_EMBEDDINGS: DataConfig(
                dependencies=[C.DATA_KEY_TEXTS],
                data_paths=[C.DIRNAME_GLOVE_EMBEDDINGS],
                data_counts=None,
                extensions=['npy'],
                precomputation_batch_keys=C.BATCH_KEY_GLOVE_EMBEDDINGS,
                precomputation_backbone_keys=C.BACKBONE_KEY_GLOVE_EMBEDDINGS,
                # Precomputation not yet supported
                precomputation_funcs=None,
                precomputation_transforms=None
            ),
            C.DATA_KEY_TEXTS: DataConfig(
                dependencies=[],
                data_paths=[C.DIRNAME_TEXTS],
                data_counts=None,
                extensions=['txt'],
                precomputation_batch_keys=C.BATCH_KEY_TEXTS,
                precomputation_backbone_keys=C.BACKBONE_KEY_TEXTS,
                precomputation_funcs=self._process_text_batch
            ),
        }

        # The problem is that some data keys are the same for raw and features, e.g. audio_spectrograms. But when the
        # features of the spectorgrams are corrupt, we do not need to check the audio data key immediatly, but first verify
        # if the raw spectrograms are ok. If we did not use the same keys here, we would also have to change the configs
        # every time we wanted to use the features. Right now, this is implicitly defined through the features
        # precomputation config.
        self.data_keys_same_for_raw_data_and_features = [
            C.DATA_KEY_AUDIO_SPECTROGRAMS,
            C.DATA_KEY_FACIAL_LANDMARKS,
            C.DATA_KEY_GLOVE_EMBEDDINGS,
            C.DATA_KEY_TEXTS
        ]
        # Populated automatically
        self._data_key_for_raw_data_key = {}
        self._raw_data_key_for_data_key = {}
        # For subsequent code to know whether a data_key was created because of an augmentation
        self._original_data_key_for_augmented_data_key = {}
        self._random_augmentation_for_augmented_data_key = {}

        # Since we have to update some of the data keys because they are the same for features and raw data, we need to
        # store the raw extension here if it differs between the two
        self._raw_extensions = {}

        self.augmentable_data_keys = [
            C.DATA_KEY_FRAMES_2D,
            C.DATA_KEY_FRAMES_3D,
            C.DATA_KEY_AUDIO_SPECTROGRAMS,
            C.DATA_KEY_TEXTS
        ]

        self.keys_to_update_to_resizing = [
            C.DATA_KEY_VIDEOS,
            C.DATA_KEY_FRAMES_2D,
            C.DATA_KEY_FRAMES_3D
        ]

        # We also update keys that are not necessarily influenced by the cropping but that
        # their number matches the number of videos, for simplicity.
        self.keys_to_update_to_cropping = [
            C.DATA_KEY_VIDEOS,
            C.DATA_KEY_FRAMES_2D,
            C.DATA_KEY_FRAMES_3D,
            C.DATA_KEY_AUDIO,
            C.DATA_KEY_AUDIO_SPECTROGRAMS,
            C.DATA_KEY_FACIAL_LANDMARKS,
            C.DATA_KEY_GLOVE_EMBEDDINGS,
            C.DATA_KEY_TEXTS
        ]

        assert set(modalities).issubset(self.AVAILABLE_MODALITIES), 'Unkown modalities requested:'\
            f'{modalities}. Please verify that the modalities are in {self.AVAILABLE_MODALITIES}.'
        self.requested_modalities = modalities
        # If we use the 2D-3D modality we have to split it into 2 modalities since, for example, for modality
        # precomputation we have individual encoders for frames_2d and frames_3d
        self._split_requested_modalities = []
        for modality in self.requested_modalities:
            if modality == C.BATCH_KEY_FRAMES_2D_3D:
                self._split_requested_modalities.extend([C.BATCH_KEY_FRAMES_2D, C.BATCH_KEY_FRAMES_3D])
            else:
                self._split_requested_modalities.append(modality)
        self.num_mels = num_mels
        self._audio_spectrograms_requested = C.DATA_KEY_AUDIO_SPECTROGRAMS in self._split_requested_modalities
        if self._audio_spectrograms_requested:
            assert num_mels is not None, f'You requested audio spectrograms but did not set the number of mels.'
        if num_mels is not None:
            assert self._audio_spectrograms_requested,\
                f'You set the number of mels but did not requested audio spectrograms.'
        self.inference_start_pos = inference_start_pos
        self.subsample_audio = subsample_audio
        self.max_embedding_length = max_embedding_length
        if max_embedding_length is not None:
            assert C.BATCH_KEY_GLOVE_EMBEDDINGS in modalities, 'If you specify ' \
                '`max_embedding_length` you have to use the glove-embedding modality (not text).'
        self.feature_precomputation_config = feature_precomputation_config
        self.num_augmented_samples_to_load = num_augmented_samples_to_load
        self._use_precomputed_features = feature_precomputation_config is not None
        self._random_augmentations = {}

        self.cache_features = cache_features
        if cache_features is not None:
            assert cache_features in ['cpu'], 'Unknown cache_features value. Please use either cpu or None.'
            assert self._use_precomputed_features, 'You set cache_features but did not use precomputed features.'

        if resize_scale is not None:
            assert isinstance(
                resize_scale, int), 'Currently only int resize scales supported.'

        # To indicate whether the paths function should return the raw path for a data key
        self._processing_raw_data = False

        self.crop_face = crop_face
        if preprocessing_batch_size is not None:
            self.preprocessing_batch_size = preprocessing_batch_size
        else:
            print('Not preprocessing_batch_size given, setting to 1.')
            self.preprocessing_batch_size = 1
        self.data_processed_dir = os.path.join(self.data_dir, C.PROCESSED_DATA_DIR, self.DATASET_NAME)
        self.resize_scale = resize_scale

        self._add_additional_data_configs()
        self._init_data_keys_same_for_raw_and_features()
        self._init_counts_for_data_keys()
        self._init_data_configs()

        if not self._use_precomputed_features:
            # In case we don't use precomputed features, we have to use padding to the videos which are of unequal length
            self._collate_fn = sequence_collate_fn

        # To cache the loaded annotations
        self._annotations = None

    def _instantiate_transforms(self, transforms: Dict):
        # We just pass as we instantiate the transforms later. There might be random augmentations
        # defined to be precomputed, therefore we cannot just instaniate the transforms here.
        pass

    @abstractmethod
    def available_modalities(self):
        # Defined by subclasses
        raise NotImplementedError()

    @abstractmethod
    def available_subsets(self):
        raise NotImplementedError()

    @abstractmethod
    def _raw_data_keys(self):
        raise NotImplementedError()

    def _add_additional_data_configs(self):
        # Can be overwritten by subclasses to append new data configs
        pass

    def _init_data_keys_same_for_raw_and_features(self):
        # Can be overwritten by subclasses to append new keys that have the same key for raw data and features and
        # thus need to be adjusted
        pass

    @abstractmethod
    def _init_counts_for_data_keys(self):
        raise NotImplementedError('Needs to be implemented by subclasses')

    def _merge_key_with_aug_name(self, key, aug_name):
        return f'{key}_{aug_name}'

    def _init_data_configs(self):
        if self._use_precomputed_features:
            # For texts, the raw extension will be different than the precomputed one, so we need to store it here
            self._raw_extensions[C.DATA_KEY_TEXTS] = self._data_configs[C.DATA_KEY_TEXTS].extensions
            self._data_configs[C.DATA_KEY_TEXTS].extensions = ['npy']
        # Append the resizing size (since that's a crucial value) and the cropping to the paths
        for key in self.keys_to_update_to_resizing:
            if self.resize_scale is not None:
                dirnames = [f'{path}_{self.resize_scale}' for path in self._data_configs[key].data_paths]
                self._data_configs[key].data_paths = dirnames
        for key in self.keys_to_update_to_cropping:
            if self.crop_face:
                dirnames = [f'{path}_cropped' for path in self._data_configs[key].data_paths]
                self._data_configs[key].data_paths = dirnames

        if self.crop_face:
            self._data_configs[C.DATA_KEY_ANNOTATIONS].data_paths = [C.FILENAME_ANNOTATIONS_CROPPED]

        if self._audio_spectrograms_requested:
            # Adjust to include the number of mels in the path
            self._data_configs[C.DATA_KEY_AUDIO_SPECTROGRAMS].data_paths = [
                f'{path}_{self.num_mels}' for path in self._data_configs[C.DATA_KEY_AUDIO_SPECTROGRAMS].data_paths]

        data_config_keys = list(self._data_configs.keys())
        for key in data_config_keys:
            data_config = self._data_configs[key]
            if key in self.data_keys_same_for_raw_data_and_features and key in self.available_modalities():
                self._add_raw_data_key(key)
            if data_config.precomputation_funcs is not None:
                data_config.data_paths = [os.path.join(C.DIRNAME_FEATURES, path) for path in data_config.data_paths]
            if self.feature_precomputation_config is not None:
                # Will initialize the transforms for the data_key. The default case is a class_path together with
                # init_args. If the transforms are a list of such entries, then it is assumed that the transforms are
                # augmentations, i.e. the data_key will be split into multiple data_keys, one for each augmentation.
                self._init_data_transforms(key, data_config)

        for key, data_config in self._data_configs.items():
            paths = data_config.data_paths
            if paths is not None:
                data_config.data_paths = [
                    os.path.join(self.data_processed_dir, path) for path in data_config.data_paths]

    def _add_raw_data_key(self, data_key):
        original_data_config = deepcopy(self._data_configs[data_key])
        # If the data key is the same for the raw data and the features we need to modify the dependency chain.
        # The reason is that e.g. spectrograms features are corrupt but the dependency chain tells us to check
        # the audio. But that is not the first thing to check, we need to verify if the raw spectrograms are
        # still ok (they are computed from the audio). This is why we copy the data from the features data
        # key to a new data key with the suffix _raw. This way, e.g. spectrograms_raw will be checked if
        # spectrograms, pointing to the features directory, are corrupt.
        raw_data_key = data_key + '_raw'
        if data_key in self._raw_extensions:
            extensions = self._raw_extensions[data_key]
        else:
            extensions = self._data_configs[data_key].extensions
        raw_data_config = DataConfig(
            dependencies=original_data_config.dependencies,
            data_paths=original_data_config.data_paths,
            data_counts=original_data_config.data_counts,
            extensions=extensions,
            precomputation_batch_keys=None,
            precomputation_backbone_keys=None,
            precomputation_funcs=None,
            precomputation_transforms=None
        )
        self._data_configs[raw_data_key] = raw_data_config
        self._data_configs[data_key].dependencies = [raw_data_key]
        # So that we can pass the proper data key to the subclasses.
        self._data_key_for_raw_data_key[raw_data_key] = data_key
        self._raw_data_key_for_data_key[data_key] = raw_data_key

    def _init_data_transforms(self, data_key, data_config):
        """ Inits the transforms for the data_key. The following formats are possible

        For using a single transform for all subsets and no repeatd augmentations:
        frames_2d:
            transforms:
                class_path: xxx
                init_args: yyy

        For using different single transforms for different subsets:
        frames_2d:
            transforms:
                train:
                    class_path: xxx
                    init_args: yyy
                val:
                    ...

        For using different augmentations for different subsets. Note that vor val and test, only single transforms are
        allowed since we do not support augmentation of test data:
        frames_2d:
            transforms:
                train:
                    - name: name1
                      class_path: xxx
                      init_args: yyy
                    - name: name2
                      class_path: zzz
                      init_args: aaa
                val:
                    class_path: xxx
                    init_args: yyy
        """
        if self.transforms_dict is None:
            return

        transforms = self.transforms_dict

        # Some small sanity checks:
        if 'class_path' in transforms:
            assert 'init_args' in transforms, 'You defined a class_path but not the init_args.'
            assert all(key in ['class_path', 'init_args'] for key in list(transforms.keys())),\
            'If you define a class_path, only init_args is allowed in addition.'

        if 'init_args' in transforms:
            assert 'class_path' in transforms, 'You defined init_args but not the class_path for the transforms.'

        if 'train' in transforms:
            if isinstance(transforms['train'], list):
                # Pass because the list is allowed and individual entries will be checked later
                pass
            elif isinstance(transforms['train'], dict):
                if 'class_path' in transforms['train']:
                    assert all(key in ['class_path', 'init_args'] for key in transforms['train'].keys()), 'If you  '\
                    'define a transform as a dict, only class_path and init_args are allowed as keys.'
                else:
                    for modality_key in transforms['train'].keys():
                        assert modality_key in self.available_modalities(), f'Unknown modality {modality_key} in '\
                        'transforms. Please check that the modality is in the available modalities.'
            else:
                raise Exception('Unknown format for train transforms.')

        transform_keys = list(transforms.keys())

        per_subset = (all(x in self.available_subsets() for x in transform_keys))
        if not per_subset:
            # Small sanity check that either the dictionary contains only subset keys or none at all, to not mix
            # something like class_path with subset keys.
            assert not (any(x in self.available_subsets() for x in transform_keys)), 'You defined some transforms '\
            'on a per-subset basis, i.e. by using train: in transforms:, but you also defined some transforms '\
            'without a subset, i.e. directly in transforms:. Please either define all transforms on a per-subset '\
            'basis or none.'
            # Transforms here is the first case mentioned in the docstring above, i.e. the same transform
            # for all subsets.
            transform_for_all_subsets = ite(caller=self, transform_tree=transforms)
            data_config.precomputation_transforms = {}
            for subset in self.available_subsets():
                data_config.precomputation_transforms[subset] = transform_for_all_subsets
        else:
            # Shared by the augmentations, if defined, and used by the normal data key.
            val_transform = ite(caller=self, transform_tree=transforms['val']) if 'val' in transform_keys else None
            test_transform = ite(caller=self, transform_tree=transforms['test']) if 'test' in transform_keys else None

            if 'train' in transform_keys:
                train_transform_cfg = transforms['train']
                if data_key in train_transform_cfg:
                    # For the case where we have e.g. frames_2d in the train transforms. For SimCLR, we don't have this
                    # but the transforms are directly set under train:.
                    train_transform_cfg = train_transform_cfg[data_key]
                # We differentiate between the case where the transforms are a list, and thus a list of
                # augmentations, and where it is a single transform (given as a dict with class_path and init_args).
                # The latter could e.g. be normalization and resizing.
                is_list_of_augmentations = isinstance(train_transform_cfg, list)
                if is_list_of_augmentations:
                    self._add_augmented_data_key(
                        data_key, data_config, train_transform_cfg, val_transform, test_transform)
                    # Doesn't matter, in _add_augmented_data_key we remove the original data key from the modalities.
                    # This makes it a bit weird since in the following code, setting the transforms on data_config will
                    # work but it will not be used.
                    train_transform = None
                else:
                    # Here we can directly instantiate the transforms because the instantiation function is capable
                    # of handling dictionaries and returns the result with the structured maintained. We cannot do this
                    # for the augmentations because there we need new augmented data keys for each augmentation.  
                    train_transform = ite(caller=self, transform_tree=train_transform_cfg)
            else:
                train_transform = None

            data_config.precomputation_transforms = {
                'train': train_transform,
                'val': val_transform,
                'test': test_transform}

    def _add_augmented_data_key(self, data_key, data_config, train_transforms_cfg, val_transforms, test_transforms):
        # If augmentations are defined, we reset the dependencies and add the augmented data keys. This
        # means that e.g. frames_2d's dependency to videos is removed and replaced by the augmented
        # frames_2d data keys, like frames_2d_aug1, frames_2d_aug2, ... and frames_2d_augX will point
        # to the videos as dependency.
        # NOTE: We set this all to empty here so that the un-augmented data will not be checked later.
        original_depenencies = data_config.dependencies
        # Remove the depenencies so that they don't point to raw data anymore. We use the normal unaugmented data keys
        # later instead of the augmented data keys, and if the dependencies were in here we would check the raw data
        # in the data checking later, even if the augmented data was all there (unecessary data processing in this case)
        data_config.dependencies = []
        original_counts = data_config.data_counts
        # Setting to None prevents any checking if the number of files is correct. This is what we want because all
        # the checking is done through the augmented keys.
        data_config.data_counts = None
        original_extensions = data_config.extensions
        data_config.extensions = []
        for random_augmentation in train_transforms_cfg:
            aug_name = random_augmentation['name']
            aug_key = self._merge_key_with_aug_name(data_key, aug_name)
            # Somehow Python is super slow when deep-copying multiple configs here, so we do it manually
            augmented_data_config = DataConfig(
                dependencies=original_depenencies,
                data_paths=[os.path.join(f'{path}_aug', aug_name) for path in data_config.data_paths],
                data_counts=original_counts,
                extensions=original_extensions,
                precomputation_batch_keys=data_config.precomputation_batch_keys,
                precomputation_backbone_keys=data_config.precomputation_backbone_keys,
                precomputation_funcs=data_config.precomputation_funcs,
                precomputation_transforms={'train': None, 'val': None, 'test': None})
            augmented_data_config.precomputation_batch_keys = aug_key
            instantiation_config = {
                'class_path': random_augmentation['class_path'],
                'init_args': random_augmentation.get('init_args', {})}
            self._random_augmentation_for_augmented_data_key[aug_key] = instantiation_config
            # Only the train transforms can be augmentations. For val and test, we only allow single transforms and
            # reuse the same transforms for all augmented keys.
            augmented_data_config.precomputation_transforms['train'] = ite(caller=self,
                                                                           transform_tree=instantiation_config)
            if val_transforms is not None:
                augmented_data_config.precomputation_transforms['val'] = val_transforms.get(data_key, NoOp())
            if test_transforms is not None:
                augmented_data_config.precomputation_transforms['test'] = test_transforms.get(data_key, NoOp())
            # For the VideoDataset, a bit hacky but I can't be arsed anymore to do it properly
            random_augmentation['dir'] = os.path.join(self.data_processed_dir, augmented_data_config.data_paths[0])
            self._data_configs[aug_key] = augmented_data_config
            #data_config.dependencies.append(aug_key)
            # Check if the user requested this modality before we add it to the modalities. We check in the split
            # modalities here so that when frames_2d_3d is requested, the individual frames_2d_augx will be added
            # even if frames_2d is not directly in self.requested_modalities
            if data_key in self._split_requested_modalities:
                self.requested_modalities.append(aug_key)
            self._original_data_key_for_augmented_data_key[aug_key] = data_key
            # Append the depency to the original data key so we know which it depends on. This is a bit weird as one
            # would expect the other way round. The reason is that e.g. the unaugmented frames_2d key will later be
            # used in the VideoDataset for the augmented ones. The dataset will only see the key frames_2d. Now, to
            # check that all the data is there, we use this dependency here. I cannot really explain it that
            # well here sorry.
            data_config.dependencies.append(aug_key)
        data_config.has_augmentations = True

    def _all_data_keys(self):
        # Here we essentially populate our config with all possible data keys that we might need. Which ones we
        # actually need to check is computed later, so not all the keys here will be verified.
        all_data_keys = [C.DATA_KEY_ANNOTATIONS]
        available_modalities = self.available_modalities()
        data_dependencies = []
        for modality in available_modalities:
            data_dependencies.extend(self._dependencies_for_data_key(modality))
        data_dependencies = list(set(data_dependencies))
        # Now we have all keys that are available/need to be checked
        all_data_keys += data_dependencies
        return all_data_keys

    def _dependencies_for_data_key(self, data_key):
        # Recursively get all dependencies for the given data_key
        dependencies = set(self._data_configs[data_key].dependencies)
        sub_dependencies = set()
        for dependency in dependencies:
            sub_dependencies.update(self._dependencies_for_data_key(dependency))
        dependencies.update(sub_dependencies)
        with_augmentations = False
        if not with_augmentations:
            # We only add the original data_key if we don't have augmentations, e.g. [frames_2d]
            dependencies.add(data_key)
        return dependencies

    def _paths_for_data_key(self, data_key):
        if self._processing_raw_data:
            if data_key in self._raw_data_key_for_data_key:
                data_key = self._raw_data_key_for_data_key[data_key]
        if data_key in self._data_configs:
            return self._data_configs[data_key].data_paths
        return None

    def _data_count_for_data_key(self, data_key):
        return self._data_configs[data_key].data_counts

    def _extension_for_data_key(self, data_key):
        return self._data_configs[data_key].extensions

    def _precomputation_batch_keys_for_data_key(self, data_key):
        return self._data_configs[data_key].precomputation_batch_keys

    def _precomputation_func_for_data_key(self, data_key):
        return self._data_configs[data_key].precomputation_funcs

    def _precomputation_transforms_for_data_key(self, data_key, subset=None):
        transforms = self._data_configs[data_key].precomputation_transforms
        if transforms is not None and transforms[subset] is not None:
            return transforms[subset]
        else:
            # For data_keys and subsets that don't have any transforms defined, we just return a NoOp. This way,
            # we don't have to differentiate between data_keys that have transforms and those that don't.
            return NoOp()

    def _precomputation_backbone_keys_for_data_key(self, data_key):
        return self._data_configs[data_key].precomputation_backbone_keys

    def _dependencies_for_data_key(self, data_key):
        return self._data_configs[data_key].dependencies

    def prepare_data(self, force_reprocess=False):
        print(f'Verifying {self.DATASET_NAME} dataset at {self.data_processed_dir}.')

        corrupt_data_keys = set()
        # Annotations need to be checked in addition, the modalities never request it directly
        requested_data_keys = set(self.requested_modalities + [C.DATA_KEY_ANNOTATIONS])
        for data_key in requested_data_keys:
            corrupt_data_keys = self._verify_data_integrity(data_key, corrupt_data_keys)

        if len(corrupt_data_keys) > 0:
            self._extract_data_and_precompute_features(corrupt_data_keys=corrupt_data_keys)

        print('Dataset verification successful.')
        if self.compute_class_weights:
            annotations_path = self._paths_for_data_key(C.DATA_KEY_ANNOTATIONS)[0]
            with open(annotations_path, 'r') as ann_file:
                annotations = yaml.safe_load(ann_file)
            self._compute_class_weights('train', annotations)
            self._compute_class_weights('val', annotations)
            self._compute_class_weights('test', annotations)

    def _verify_data_integrity(self, data_key, corrupt_data_keys=set(), force_reprocess=False):
        # Quick exit so that we don't check dependencies multiple times
        if data_key in corrupt_data_keys:
            return corrupt_data_keys
        counts = self._data_count_for_data_key(data_key)
        has_counts = counts is not None
        # Some configs, like the one for frames_2d_3d, don't have the paths directly on them but only in the subkeys
        if has_counts:
            paths = self._paths_for_data_key(data_key)
            extensions = self._extension_for_data_key(data_key)
            for path, count, extension in zip(paths, counts, extensions):
                precomputation_func = self._precomputation_func_for_data_key(data_key)
                self._verify_dir_or_file_integrity(
                    data_key, precomputation_func, corrupt_data_keys, path, count, extension, force_reprocess)
        # Only check if the dependencies are correct if the current data is corrupt. This way, we avoid having
        # to e.g. extract all the videos again if the precomputed features, that were actually requested, exist already.
        data_corrupt = data_key in corrupt_data_keys
        dependencies = self._dependencies_for_data_key(data_key)
        has_dependencies = dependencies is not None
        # If the data is corrupt or we don't have counts (e.g. for frames_2d_3d), we need to check the dependencies
        if (data_corrupt and has_dependencies) or (not has_counts and has_dependencies):
            for dependency in dependencies:
                self._verify_data_integrity(dependency, corrupt_data_keys)
        return corrupt_data_keys

    def _verify_dir_or_file_integrity(
            self, data_key, precomputation_func, corrupt_data_keys, path, count, extension, force_reprocess):
        is_file = path.endswith(extension)
        if is_file:
            if not file_util.filesize_ok(path, count) or force_reprocess:
                corrupt_data_keys.add(data_key)
        else:
            if not file_util.num_files_ok(path, count, extension) or force_reprocess:
                corrupt_data_keys.add(data_key)
            elif precomputation_func is not None:
                # Check if the precomputation config is still the same
                precomp_config_file_path = os.path.join(path, C.FILENAME_PRECOMP_CONFIG)
                if not os.path.exists(precomp_config_file_path):
                    print(f'Precomputation config file {precomp_config_file_path} does not exist. '
                        'Need to recompute features.')
                    corrupt_data_keys.add(data_key)
                else:
                    with open(precomp_config_file_path, 'r') as precomp_config_file:
                        existing_precomp_config = yaml.safe_load(precomp_config_file)
                        config_to_compare = self._get_precomp_config_to_store(data_key)
                        existing_dumped_config = yaml.safe_dump(existing_precomp_config)
                        expected_dumped_config = yaml.safe_dump(config_to_compare)
                        if existing_dumped_config != expected_dumped_config:
                            print(f'Precomputation config file {precomp_config_file_path} does not match '
                                'the current model backbone config:')
                            print(f'Expected config:')
                            print(expected_dumped_config)
                            print(f'Existing config:')
                            print(existing_dumped_config)
                            print('Need to recompute features.')
                            corrupt_data_keys.add(data_key)

    def _get_precomp_config_to_store(self, data_key):
        # This is the case if we have an augmented data key, e.g. frames_2d_aug1. We also want to write out
        # the augmentations in this case and we need the original key to retrieve the backbone config.
        original_data_key = self._original_data_key_for_augmented_data_key.get(data_key)

        key = data_key if original_data_key is None else original_data_key

        if key in self._model_backbone_config:
            # This is the case if the model config defines a backbone on a per-modality basis,
            # as the ConCluModel does for example. The SimCLR model directly defines the backbone,
            # and not as a child of a backbone_key.
            config_to_compare = copy(self._model_backbone_config[key])
        else:
            # Here we get the config that we store in the features directories to be able to identify whether something
            # in the way the features are to be computed has changed.
            config_to_compare = copy(self._model_backbone_config)

        if key in self.feature_precomputation_config:
            config_to_compare.update(self.feature_precomputation_config[key])

        if data_key in self._random_augmentation_for_augmented_data_key:
            if 'transforms' not in config_to_compare:
                config_to_compare['transforms'] = {}
            config_to_compare['transforms']['train'] = self._random_augmentation_for_augmented_data_key[data_key]
            if 'val' in self.transforms_dict:
                if 'class_path' in self.transforms_dict['val']:
                    config_to_compare['transforms']['val'] = self.transforms_dict['val']
                elif data_key in self.transforms_dict['val']:
                    config_to_compare['transforms']['val'] = self.transforms_dict['val'][data_key]
            if 'test' in self.transforms_dict:
                if 'class_path' in self.transforms_dict['test']:
                    config_to_compare['transforms']['test'] = self.transforms_dict['test']
                elif data_key in self.transforms_dict['test']:
                    config_to_compare['transforms']['test'] = self.transforms_dict['test'][data_key]
        else:
            one_transform_for_all_subsets = self.transforms_dict is not None and\
                all([subset not in self.transforms_dict for subset in self.available_subsets()])
            for subset in self.available_subsets():
                transforms = self.transforms_dict.get(subset)
                to_assign = None
                # If getting the subset is None, then there is one transform defined directly for all subsets
                # or self.transforms_dict is None
                if transforms is not None:
                    if 'class_path' in transforms:
                        to_assign = transforms
                    elif data_key in transforms:
                        to_assign = transforms[data_key]
                else:
                    if one_transform_for_all_subsets:
                        if 'class_path' in self.transforms_dict:  
                            to_assign = self.transforms_dict
                        elif data_key in self.transforms_dict:
                            to_assign = self.transforms_dict[data_key]
                if to_assign is not None:
                    if 'transforms' not in config_to_compare:
                        config_to_compare['transforms'] = {}
                    if subset not in config_to_compare['transforms']:
                        config_to_compare['transforms'][subset] = {}
                    config_to_compare['transforms'][subset] = to_assign

        return config_to_compare

    @abstractmethod
    def _extract_raw_data(self, corrupt_data_configs):
        raise NotImplementedError()

    def _extract_data_and_precompute_features(self, corrupt_data_keys):
        corrupt_paths = set()
        for data_key in corrupt_data_keys:
            paths_to_add = [
                (path, path.endswith(extension), data_key) for
                path, extension in zip(self._paths_for_data_key(data_key), self._extension_for_data_key(data_key))]
            corrupt_paths.update(set(paths_to_add))

        self._remove_corrupt_paths(corrupt_paths)

        if bool(set(self._raw_data_keys()) & set(corrupt_data_keys)):
            print('Extracting raw data...', flush=True)
            self._extract_raw_data(corrupt_data_keys)

        # Configs that don't have a precomputation func are plain data configs, i.e. they don't need to
        # be passed to the precomputation function and, more importantly, should not be passed to the
        # subclasses extract_data functions. Such configs are e.g. frames_2d.
        corrupt_plain_data_configs = []
        for data_key in corrupt_data_keys:
            if self._precomputation_func_for_data_key(data_key) is None:
                corrupt_plain_data_configs.append(data_key)
        if len(corrupt_plain_data_configs) > 0:
            # There is a problem with the precomputed features keys as they need to reference their raw data. E.g.,
            # in the case of the spectrograms, this is the spectrograms again. That is why our init adds some temporary
            # raw keys to the data configs. We need to map them back to the original keys here so that the subclasses
            # can handle them.
            sanitized_corrupt_plain_data_configs = []
            for data_key in corrupt_plain_data_configs:
                if data_key in self._data_key_for_raw_data_key:
                    sanitized_corrupt_plain_data_configs.append(self._data_key_for_raw_data_key[data_key])
                else:
                    sanitized_corrupt_plain_data_configs.append(data_key)
            # If there is any raw modality to be processed, we need to add the raw videos key since the video files
            # are the most elementary data. The subclasses use these files to iterate over the data to be processed.
            if C.DATA_KEY_VIDEOS_ORIGINAL not in sanitized_corrupt_plain_data_configs:
                sanitized_corrupt_plain_data_configs.append(C.DATA_KEY_VIDEOS_ORIGINAL)
            self._processing_raw_data = True
            self._extract_data(sanitized_corrupt_plain_data_configs)
            self._processing_raw_data = False

        corrupt_feature_precomputation_data_configs = []
        for data_key in corrupt_data_keys:
            if self._precomputation_func_for_data_key(data_key) is not None:
                corrupt_feature_precomputation_data_configs.append(data_key)
        if len(corrupt_feature_precomputation_data_configs) > 0:
            self._precompute_features(corrupt_feature_precomputation_data_configs)

    def _remove_corrupt_paths(self, corrupt_paths):
        remove_all = False
        remove_none = False
        if len(corrupt_paths) > 1:
            choose_all_text = f'Found {len(corrupt_paths)} corrupt paths. Do you want to remove all (A) or choose '\
                               'individually (i) or none (n)? (A/i/n) '
            choice = input_util.get_choice(choose_all_text, ['A', 'i', 'n'])
            if choice == 'A':
                remove_all = True
                print('Removing all corrupt paths.')
            elif choice == 'n':
                remove_none = True
            elif choice == 'i':
                # Will allow the user to select which path to remove individually in the following
                pass

        if not remove_none:
            for path, is_file, data_key in corrupt_paths:
                if os.path.exists(path):
                    if not remove_all:
                        text = f'Do you want to remove the corrupt path {path}? (Y/n) '
                        choice = input_util.get_choice(text, ['Y', 'n'])
                    else:
                        choice = 'Y'
                    if choice == 'Y':
                        print(f'Removing corrupt path {path}.')
                        if is_file:
                            os.remove(path)
                        else:
                            file_util.rmdir(path)
                    else:
                        print(f'Not removing path {path}.')
                else:
                    print(f'Path {path} does not exist (will be created).')

        for path, is_file, data_key in corrupt_paths:         
            # We now recreate the data paths for subsequent code
            if not is_file:
                for available_subset in self.available_subsets():
                    os.makedirs(os.path.join(path, available_subset), exist_ok=True)

    def _extract_data(self, corrupt_data_configs):
        raise NotImplementedError()

    def _extract_audio_spectrogram(self, input_file_path, output_file_path):
        # Helper function for the subclasses so that the audio always gets extracted in the same way
        audio_util.extract_spectrogram_from_audio_file(
            input_file_path,
            output_file_path,
            sample_rate=C.AUDIO_SAMPLE_RATE,
            hop_length=C.AUDIO_HOP_LENGTH,
            n_fft=C.AUDIO_N_FFT,
            n_mels=self.num_mels
        )

    def _precompute_features(self, corrupt_data_configs: dict):
        global sig_int_received
        print('Precomputing features...', flush=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _feature_extractors = {}

        for data_key in corrupt_data_configs:
            # Some models don't define the backbone as the value to the modalities key. In this case, we
            # simply store the backbone config directly.
            extractor_config = self._model_backbone_config
            backbone_key = self._precomputation_backbone_keys_for_data_key(data_key)
            if backbone_key in self._model_backbone_config:
                extractor_config = self._model_backbone_config[backbone_key]
            if backbone_key not in _feature_extractors:
                extractor = backbone_loader.load_backbone(extractor_config)
                extractor.eval()
                extractor.to(device)
                _feature_extractors[backbone_key] = extractor
            for path in self._paths_for_data_key(data_key):
                precomp_config_file_path = os.path.join(path, C.FILENAME_PRECOMP_CONFIG)
                config_to_store = self._get_precomp_config_to_store(data_key)
                os.makedirs(path, exist_ok=True)
                with open(precomp_config_file_path, 'w+') as precomp_config_file:
                    yaml.safe_dump(config_to_store, precomp_config_file)

        # Annotations file is always only one path
        annotations_file_path = self._paths_for_data_key(C.DATA_KEY_ANNOTATIONS)[0]
        with open(annotations_file_path, 'r') as annotations_file:
            annotations = yaml.safe_load(annotations_file)
        annotations = self._prepare_annotations_data(annotations)

        if C.PRECOMP_KEY_FRAMES_2D in self.feature_precomputation_config:
            frames_2d_target_fps = self.feature_precomputation_config[
                C.PRECOMP_KEY_FRAMES_2D][C.PRECOMP_KEY_TARGET_FPS]
        else:
            frames_2d_target_fps = None
        if C.PRECOMP_KEY_FRAMES_3D in self.feature_precomputation_config:
            frames_3d_target_fps = self.feature_precomputation_config[
                C.PRECOMP_KEY_FRAMES_3D][C.PRECOMP_KEY_TARGET_FPS]
        else:
            frames_3d_target_fps = None
        text_model = None
        if C.BACKBONE_KEY_TEXTS in self._model_backbone_config:
            text_model = self._model_backbone_config[C.BACKBONE_KEY_TEXTS][C.BACKBONE_KEY_HUGGINFACE_MODEL]

        if self.preprocessing_batch_size > 32:
            print(f'WARNING: The preprocessing batch size is large ({self.preprocessing_batch_size}). '
                  'This might lead to memory issues.')

        modalities = list(corrupt_data_configs)

        process_range = self.feature_precomputation_config.get('process_range')

        paths_for_modality = {}
        # The raw data keys are the ones for keys that have the same key for original and augmented key, like
        # audio_spectrograms, and we add manually the videos key for frames_2d_3d. _split_modalities contain some
        # unecessary keys like frames_2d, but we can just ignore that and are more flexible like this.
        modalities_to_check = self._split_requested_modalities + [C.BATCH_KEY_VIDEOS]
        for data_key in modalities_to_check:
            if data_key in self._raw_data_key_for_data_key:
                raw_data_key = self._raw_data_key_for_data_key[data_key]
                paths = self._paths_for_data_key(raw_data_key)
            else:
                paths = self._paths_for_data_key(data_key)
            paths_for_modality[data_key] = paths[0]

        for subset in self.available_subsets():
            print(f'Processing subset {subset}...', flush=True)
            transforms = {k: self._precomputation_transforms_for_data_key(k, subset) for k in modalities}
            dataset = DynamicVideoClassificationDataset(
                data_dir=self.data_processed_dir,
                subset=subset,
                paths_for_modality=paths_for_modality,
                annotations=annotations[subset],
                # We first process the frames only and so on
                modalities=modalities,
                multi_label=self.multi_label,
                transforms=transforms,
                frames_2d_target_fps=frames_2d_target_fps,
                frames_3d_target_fps=frames_3d_target_fps,
                hugginface_tokenizer=text_model)
            sampler = None
            if process_range is not None:
                subset_process_range = process_range[subset] if subset in process_range else process_range
                if isinstance(subset_process_range, int):
                    print(f'Precomputing features from index {subset_process_range} to end.')
                    sampler = RangeSampler(dataset, subset_process_range)
                else:
                    print(
                        f'Prcomputing features from index {subset_process_range[0]} to index {subset_process_range[1]}.')
                    sampler = RangeSampler(dataset, subset_process_range[0], subset_process_range[1])
            # Using self.batch_size should always work since we are only using one modality here
            if sampler is None:
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.preprocessing_batch_size,
                    shuffle=False,
                    num_workers=self.num_cpus,
                    collate_fn=sequence_collate_fn)
            else:
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.preprocessing_batch_size,
                    sampler=sampler,
                    num_workers=self.num_cpus,
                    collate_fn=sequence_collate_fn)
            start_batch = time.time()

            # Process the batches
            for i, batch in enumerate(dataloader):
                try:
                    self._process_batch(i, dataloader, modalities, batch, device,
                                        _feature_extractors, subset, start_batch)
                    start_batch = time.time()
                except KeyboardInterrupt:
                    print('SIGINT received, stopping preprocessing.', flush=True)
                    break

        # Clear the GPU memory
        for key in self._split_requested_modalities:
            if key in _feature_extractors:
                _feature_extractors[key].to('cpu')
        del _feature_extractors

    def _process_batch(self, i, dataloader, modalities, batch, device, _feature_extractors, subset, start_batch):
        print(f'Processing batch {i + 1}/{len(dataloader)} (obtained in {time.time() - start_batch:.2f}s)',
                flush=True)
        start_extract = time.time()
        for modality in modalities:
            if len(modalities) == 1:
                batch_data_for_modality = batch[C.BATCH_KEY_INPUTS]
                batch_filenames_for_modality = batch[C.BATCH_KEY_FILENAMES]
            else:
                batch_key = self._precomputation_batch_keys_for_data_key(modality)
                batch_data_for_modality = batch[C.BATCH_KEY_INPUTS][batch_key]
                batch_filenames_for_modality = batch[C.BATCH_KEY_FILENAMES][batch_key]
            process_func = self._precomputation_func_for_data_key(modality)
            out_dir = [os.path.join(path, subset) for path in self._paths_for_data_key(modality)]
            # Currently there is only one out dir for all
            out_dir = out_dir[0]
            process_func(
                batch_data_for_modality, batch_filenames_for_modality, device, out_dir, _feature_extractors)
            torch.cuda.empty_cache()
        print(f'Extracting features took {time.time() - start_extract:.2f}s', flush=True)

    def _resolve_raw_dir(self, data_key):
        raw_data_key = self._raw_data_key_for_data_key.get(data_key)
        if raw_data_key is not None:
            return self._paths_for_data_key(raw_data_key)[0]
        return None

    def _process_video_2d_batch(self, batch, filenames, device, out_dir, _feature_extractors):
        # frames is [batch_size, num_frames, channels, height, width]
        with torch.no_grad():
            extractor = _feature_extractors[C.PRECOMP_KEY_FRAMES_2D]
            # If the number of extracted frames is not different for the samples extracted into the batch,
            # we receive a tensor directly and not a PackedSequence. That's why we have to differentiate here.
            if isinstance(batch, PackedSequence):
                frames, lengths = pad_packed_sequence(
                    batch, batch_first=True)
            else:
                frames = batch
                lengths = None
            # frames is of shape [batch_size, num_frames, channels, height, width] but 2D networks can only
            # handle [batch_size, channels, height, width], that's why we process each batch entry individually
            # and num_frames is treated as the batch size. At the same time we can limit the number of frames
            # to the unpadded frames
            frames = frames.to(device)
            # all_features will be a list of entries like [num_frames, num_features]
            if lengths is not None:
                all_features = [extractor(frames[i][:lengths[i]])[C.KEY_RESULTS_FEATURES].detach() for i in
                                range(len(frames))]
            else:
                all_features = [extractor(frames[i])[C.KEY_RESULTS_FEATURES].detach() for i in range(len(frames))]
            for i, filename in enumerate(filenames):
                # Features is of shape [num_frames, num_features]
                features = all_features[i]
                mean_feature = features.mean(0)
                identifier = os.path.basename(filename).replace('.mp4', '')
                output_video_features_2d_file_path = os.path.join(
                    out_dir, f'{identifier}.npy')
                # print(
                #    f'Saving 2D video features to {output_video_features_2d_file_path}', flush=True)
                np.save(output_video_features_2d_file_path, mean_feature.cpu().numpy())

    def _process_video_3d_batch(self, batch, filenames, device, out_dir, _feature_extractors):
        with torch.no_grad():
            extractor = _feature_extractors[C.PRECOMP_KEY_FRAMES_3D]
            # If the number of extracted frames is not different for the samples extracted into the batch,
            # we receive a tensor directly and not a PackedSequence. That's why we have to differentiate here.
            if isinstance(batch, PackedSequence):
                frames, lengths = pad_packed_sequence(
                    batch, batch_first=True)
            else:
                frames = batch
                lengths = None
            frames = frames.to(device)
            # frames is of shape [batch_size, num_frames, channels, height, width] but Pytorch expects
            # [batch_size, channels, num_frames, height, width]
            frames = frames.permute(0, 2, 1, 3, 4)
            num_frames = frames.shape[2]
            window_size = self.feature_precomputation_config[C.PRECOMP_KEY_FRAMES_3D][C.PRECOMP_KEY_WINDOW_SIZE]
            window_stride = self.feature_precomputation_config[C.PRECOMP_KEY_FRAMES_3D][C.PRECOMP_KEY_WINDOW_STRIDE]
            if num_frames < window_size:
                window_size = num_frames
            current_frame_pos = 0
            outputs = []
            while current_frame_pos < num_frames:
                end = min(current_frame_pos + window_size, num_frames)
                # window is of shape (batch_size, channels, window_size, height, width)
                window = frames[:, :, current_frame_pos:end]
                features = extractor(window)[C.KEY_RESULTS_FEATURES].detach()
                outputs.append(features)
                current_frame_pos += window_stride
            outputs = torch.stack(outputs)
            for i, filename in enumerate(filenames):
                # outputs is of shape [num_windows, batch_size, num_features], that's why we index the second dimension
                features = outputs[:, i]
                if lengths is not None:
                    # There are some windows which contain some padded frames at the end, that's why we take math.ceil.
                    # We divide the lenght of the current sample (lengths[i]) by the window stride to get the start
                    # of the purley padded windows
                    padding_start = math.ceil(lengths[i] / window_stride)
                    # Remove the features that were produced by purely padded frames and then take the mean over time
                    features = features[:padding_start].mean(0)
                else:
                    features = features.mean(0)
                identifier = os.path.basename(filename).replace('.mp4', '')
                output_video_features_3d_file_path = os.path.join(
                    out_dir, f'{identifier}.npy')
                # print(
                #    f'Saving 3D video features to {output_video_features_3d_file_path}', flush=True)
                np.save(output_video_features_3d_file_path, features.cpu().numpy())

    def _process_specrograms_batch(self, batch, filenames, device, out_dir, _feature_extractors):
        with torch.no_grad():
            extractor = _feature_extractors[C.PRECOMP_KEY_SPECTROGRAMS]
            if isinstance(batch, PackedSequence):
                # We don't need the lengths of the padded data because we cannot remove the padded features anyways
                spectrograms, lengths = pad_packed_sequence(
                    batch, batch_first=True)
            else:
                spectrograms = batch
                lengths = None
            # Spectrograms is of shape [batch_size, num_frames, num_mels] (that's what the VideoDataset gives us)
            spectrograms = spectrograms.to(device)
            # Permute to [batch_size, num_mels, num_frames]
            spectrograms = spectrograms.permute(0, 2, 1)
            if lengths is not None:
                all_features = [
                    extractor(spectrograms[i][:, :lengths[i]].unsqueeze(0)).detach()
                    for i in range(len(spectrograms))]
            else:
                all_features = [
                    extractor(spectrograms[i].unsqueeze(0)).detach()
                    for i in range(len(spectrograms))]
            # all_features is of shape [batch_size, num_features, num_frames]
            for i, filename in enumerate(filenames):
                # features is of shape [num_features, num_frames]
                features: torch.Tensor = all_features[i]
                # Remove leading batch dimension of size 1
                features = features.squeeze()
                # Take the mean over time, dim=-1 is the last dimension which is the
                # time dimension for audio specrograms
                features = features_util.average_into_first_dim(features)
                # We can use `filename` directly because unlike the videos which have mp4 as extension, the
                # spectrograms already have npy, just like the features we are going to save
                spectogram_features_file_path = os.path.join(
                    out_dir, filename)
                # print(
                #    f'Saving spectrogram features to {spectogram_features_file_path}', flush=True)
                np.save(spectogram_features_file_path, features.cpu().numpy())

    def _process_text_batch(self, batch, filenames, device, out_dir, _feature_extractors):
        with torch.no_grad():
            extractor = _feature_extractors[C.PRECOMP_KEY_TEXTS]
            texts = {}
            lengths = None
            if len(batch) == 2:
                data_keys = ['input_ids', 'attention_mask']
            else:
                data_keys = ['input_ids', 'token_type_ids', 'attention_mask']
            for i, data_key in enumerate(data_keys):
                batch_data = batch[i]
                if isinstance(batch_data, PackedSequence):
                    # We don't need the lengths of the padded data because we couldn't remove the padded features anyways
                    # and the attention masks already somewhat do that
                    data, _lengths = pad_packed_sequence(
                        batch_data, batch_first=True)
                    # We simply take the lenghts of the first entry, i.e. input_ids since the rest will be the same
                    if lengths is None:
                        lengths = _lengths
                else:
                    data = batch_data
                    lengths = None
                texts[data_key] = data.to(device)
            if lengths is not None:
                # A bit of a crayz thing, the extractor needs a dictionary with the keys from above and at the same
                # time we want to index into the tensors with the lengths.
                all_features = [extractor({k: v[i][:lengths[i]].unsqueeze(0) for k, v in texts.items()}).detach() for i in
                                range(len(lengths))]
            else:
                all_features = extractor(texts).detach()
            for i, filename in enumerate(filenames):
                identifier = os.path.basename(filename).replace('.txt', '')
                features = all_features[i]
                features = features.squeeze()
                if len(features.shape) == 2:
                    # We max pool over the words if the transformer returns a time dimension
                    features = torch.max(features, 1)[0]
                text_features_file_path = os.path.join(out_dir, f'{identifier}.npy')
                # print(f'Saving text features to {text_features_file_path}', flush=True)
                np.save(text_features_file_path, features.cpu().numpy())

    def _prepare_annotations_data(self, annotations):
        # Can be overwritten by subclasses to modify the annotations
        return annotations

    def test_subset_name(self):
        # Weird hack, but e.g. for RAVDESS there is no actual val and test datasets. We have the augmented train samples
        # and the unaugmented val samples. But to not duplicate even more we don't have test samples. Therfore, in
        # the RAVDESS datamodule we can manually set this here to 'val' so that 'val' will be passed to the VideoDataset
        # instead of 'test' and it will load the val samples test.
        return 'test'

    def setup(self, stage: str = None):
        if self._annotations is None:
            annotations_path = self._paths_for_data_key(C.DATA_KEY_ANNOTATIONS)[0]
            with open(annotations_path, 'r') as annotations_file:
                annotations = yaml.safe_load(annotations_file)
            self._annotations = self._prepare_annotations_data(annotations)
        
        paths_for_modality = {}
        modalities_to_check = self._split_requested_modalities
        for data_key in modalities_to_check:
            data_config = self._data_configs[data_key]
            if data_config.has_augmentations:
                paths = []
                for dependency in data_config.dependencies:
                    assert len(self._paths_for_data_key(dependency)) == 1
                    paths.append(self._paths_for_data_key(dependency)[0])
            else:
                paths = self._paths_for_data_key(data_key)
            if len(paths) > 1:
                paths_for_modality[data_key] = paths
            else:
                paths_for_modality[data_key] = paths[0]

        dataset_args = {
            'target_annotation': self.target_annotation,
            'data_dir': self.data_processed_dir,
            'modalities': self.requested_modalities,
            'split_modalities': self._split_requested_modalities,
            # Multilabel will come from the model's config to our init here
            'multi_label': self.multi_label,
            # Will only be used to check if the data key has random augmentations stored on disk
            'paths_for_modality': paths_for_modality,
            'num_augmentated_samples_to_load': self.num_augmented_samples_to_load,
        }
        if self.cache_features is not None:
            dataset_args['cache_features'] = self.cache_features

        if not self._use_precomputed_features:
            dataset_class = DynamicVideoClassificationDataset
            if C.BATCH_KEY_FRAMES_2D in self.requested_modalities or C.BATCH_KEY_FRAMES_2D_3D in self.requested_modalities:
                frames_2d_target_fps = self._model_backbone_config[C.BACKBONE_KEY_FRAMES_2D][C.PRECOMP_KEY_TARGET_FPS]
            else:
                frames_2d_target_fps = None
            if C.BATCH_KEY_FRAMES_3D in self.requested_modalities or C.BATCH_KEY_FRAMES_2D_3D in self.requested_modalities:
                frames_3d_target_fps = self._model_backbone_config[C.BACKBONE_KEY_FRAMES_3D][C.PRECOMP_KEY_TARGET_FPS]
            else:
                frames_3d_target_fps = None
            if C.BATCH_KEY_TEXTS in self.requested_modalities:
                hugginfacetokenizer = self._model_backbone_config[C.BACKBONE_KEY_TEXTS][C.BACKBONE_KEY_HUGGINFACE_MODEL]
            else:
                hugginfacetokenizer = None
            # In this case we don't have any precomputed features
            dataset_args.update({
                'frames_2d_target_fps': frames_2d_target_fps,
                'frames_3d_target_fps': frames_3d_target_fps,
                'hugginface_tokenizer': hugginfacetokenizer,
            })
            if stage == 'fit':
                dataset_args['transforms'] = self.transforms.get(
                    'train', self.transforms) if isinstance(self.transforms, dict) else self.transforms
            elif stage == 'validate':
                dataset_args['transforms'] = self.transforms.get(
                    'val', self.transforms) if isinstance(self.transforms, dict) else self.transforms
            elif stage == 'test':
                dataset_args['transforms'] = self.transforms.get(
                    'test', self.transforms) if isinstance(self.transforms, dict) else self.transforms
        else:
            dataset_class = PrecomputedFeaturesVideoClassificationDataset
            dataset_args['transforms'] = None

        if not stage or stage == 'fit':
            dataset_args['subset'] = 'train'
            dataset_args['annotations'] = self._annotations['train']
            self.train_dataset = dataset_class(**dataset_args)
        # If the stage is not "validate", we only need to create the validation dataset if the trainer is configured
        # to do validation sanity checks. There are datasets (like VoxCeleb2) which don't have a validation data set
        # and we cannot set up any.
        fitting_or_stage_none_and_do_sanity_check = (stage == 'fit' or not stage) and \
                                                    self.trainer.num_sanity_val_steps > 0
        if (stage == 'validate' or fitting_or_stage_none_and_do_sanity_check):
            dataset_args['subset'] = 'val'
            dataset_args['annotations'] = self._annotations['val']
            self.val_dataset = dataset_class(**dataset_args)

        if not stage or stage == 'test':
            dataset_args['subset'] = self.test_subset_name()
            dataset_args['annotations'] = self._annotations['test']
            self.test_dataset = dataset_class(**dataset_args)


class VideoClassificationDataModule(VideoBaseDataModule, ClassificationDataModule):
    """DataModule for video classification tasks. The only purpose of this class
    is to combine the VideoBaseDataModule and ClassificationDataModule."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
