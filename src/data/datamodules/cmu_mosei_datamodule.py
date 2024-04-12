import collections
import copy
import os
import shutil
import subprocess
import random
from typing import List

import wget
import cv2
import numpy as np
import yaml
from moviepy.editor import VideoFileClip
import torchaudio
import torch
import torch.nn as nn
from pytorch_lightning import Trainer

from data.datamodules import DataConfig
from data.datamodules import VideoClassificationDataModule
from mmsdk import mmdatasdk
from utils import constants as C
from utils import file_util, video_util, audio_util, crop_face_util

DATA = ['highlevel', 'labels', 'raw']  # ['highlevel', 'labels', 'raw']

DATA_SDK_KEYS = {
    'highlevel': mmdatasdk.cmu_mosei.highlevel,
    'labels': mmdatasdk.cmu_mosei.labels,
    'raw': mmdatasdk.cmu_mosei.raw
}

EXTERNAL_FILES = {
    'highlevel': [
        ('CMU_MOSEI_COVAREP.csd', 11585596442),
        ('CMU_MOSEI_TimestampedWordVectors.csd', 1550778982),
        ('CMU_MOSEI_VisualFacet42.csd', 1656499074),
        ('CMU_MOSEI_VisualOpenFace2.csd', 16718477556)
    ],
    'labels': [
        ('CMU_MOSEI_Labels.csd', 23254352)
    ],
    'raw': [
        ('CMU_MOSEI_TimestampedPhones.csd', 59563650),
        ('CMU_MOSEI_TimestampedWords.csd', 37943951)
    ]
}

# To be able to identify the keys which we have to remove from the recipe for the data SDK
FILENAMES_TO_SDK_KEYS = {
    'CMU_MOSEI_TimestampedWordVectors.csd': 'glove_vectors',
    'CMU_MOSEI_COVAREP.csd': 'COVAREP',
    'CMU_MOSEI_VisualOpenFace2.csd': 'OpenFace_2',
    'CMU_MOSEI_VisualFacet42.csd': 'FACET 4.2',
    'CMU_MOSEI_Labels.csd': 'All Labels',
    'CMU_MOSEI_TimestampedWords.csd': 'words',
    'CMU_MOSEI_TimestampedPhones.csd': 'phones'
}

FILES_TO_DOWNLOAD = {
    'CMU_MOSEI.zip': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI.zip'
}

FILES_TO_DOWNLOAD_SIZES = {
    'CMU_MOSEI.zip': 126081363749
}

COMBINED_VIDEO_DIR = os.path.join('Raw', 'Videos', 'Full', 'Combined')


class CMUMOSEIVideoDataModule(VideoClassificationDataModule):

    AVAILABLE_SUBSETS = ['train', 'val', 'test']

    AVAILABLE_MODALITIES = [
        C.BATCH_KEY_FRAMES_2D_3D,
        C.BATCH_KEY_FRAMES_2D,
        C.BATCH_KEY_FRAMES_3D,
        C.BATCH_KEY_AUDIO_SPECTROGRAMS,
        C.BATCH_KEY_FACIAL_LANDMARKS,
        C.BATCH_KEY_GLOVE_EMBEDDINGS,
        C.BATCH_KEY_TEXTS]

    DATASET_NAME = C.DATASET_CMU_MOSEI

    def __init__(
        self,
        label_type: str = 'continuous',
        label_discretization_threshold: float = None,
        add_neutral_label: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert label_type in ['continuous', 'discrete'], 'Only `continuous and '\
            'discrete label type supported.'
        self.label_type = label_type
        self.label_discretization_threshold = label_discretization_threshold
        if self.label_discretization_threshold is not None:
            assert self.label_type == 'discrete', 'If you pass a discretization threshold '\
                'you need to set `label_type` to `discrete`.'
        if self.label_type == 'discrete':
            assert self.label_discretization_threshold is not None, 'If you pass `label_type` '\
                'as `discrete` you need to provide a discretization threshold.'
        self.add_neutral_label = add_neutral_label
        if self.add_neutral_label:
            assert self.label_type == 'discrete', 'If you want to add a neutral label you need '\
                'to set `label_type` to `discrete`.'
            assert self.num_classes == 7, 'If you want to add a neutral label you need to have '\
                '7 classes. Did you forget to add the neutral label to the list of labels? '\
                'Or did you choose the wrong data config?'
        self.data_external_dir = os.path.join(
            self.data_dir, C.EXTERNAL_DATA_DIR, C.DATASET_CMU_MOSEI)
        self.data_external_videos_dir = os.path.join(
            self.data_external_dir, 'RAW', 'Videos', 'Full', 'Combined')

    def available_modalities(self):
        return self.AVAILABLE_MODALITIES

    def available_subsets(self):
        return self.AVAILABLE_SUBSETS

    def _raw_data_keys(self):
        return [C.DATA_KEY_VIDEOS_ORIGINAL]

    def _add_additional_data_configs(self):
        self._data_configs[C.DATA_KEY_VIDEO_INTERVALS] = DataConfig(
                dependencies=[C.DATA_KEY_VIDEOS_ORIGINAL],
                data_paths=[C.DIRNAME_VIDEO_INTERVALS],
                data_counts=[22860],
                extensions=['mp4'],
                precomputation_batch_keys=None,
                precomputation_backbone_keys=None,
                precomputation_funcs=None,
                precomputation_transforms=None
        )
        self._data_configs[C.DATA_KEY_VIDEOS].dependencies = [C.DATA_KEY_VIDEO_INTERVALS]

    def _init_counts_for_data_keys(self):
        # Counts for uncropped videos. Cropping the faces discards some of the videos.
        counts = 22860
        annotations_file_size = 3968715
        if self.crop_face:
            counts = 22853
            annotations_file_size = 4612485

        # We also set the varying count for the audio spectrograms etc., since only the audio spectrograms that
        # match a video file are being processed. Since some videos don't have a cropped counterpart (since e.g.
        # no face was detected), the number of audio spectrograms is also smaller.
        # If we ever used the uncropped videos again, this would be a problem, because the number woulnd't be the
        # one we define in the dict here and the dataset would get reprocessed.
        # But since we will always just use the cropped version that's ok.
        update_dict = {
            C.DATA_KEY_VIDEOS_ORIGINAL: [3837],
            C.DATA_KEY_VIDEOS: [counts],
            C.DATA_KEY_FRAMES_2D_3D: None,
            C.DATA_KEY_FRAMES_2D: [counts],
            C.DATA_KEY_FRAMES_3D: [counts],
            C.DATA_KEY_AUDIO: [counts],
            C.DATA_KEY_AUDIO_SPECTROGRAMS: [counts],
            C.DATA_KEY_FACIAL_LANDMARKS: [counts],
            C.DATA_KEY_GLOVE_EMBEDDINGS: [counts],
            C.DATA_KEY_TEXTS: [counts],
            C.DATA_KEY_ANNOTATIONS: [annotations_file_size]
        }

        for key, count in update_dict.items():
            self._data_configs[key].data_counts = count

    def _extract_raw_data(self, corrupt_data_configs):
        file_to_download = 'CMU_MOSEI.zip'
        size = FILES_TO_DOWNLOAD_SIZES[file_to_download]
        external_file_path = os.path.join(
            self.data_external_dir, file_to_download)
        if not os.path.exists(external_file_path) or os.stat(external_file_path).st_size != size:
            raise Exception('CMU_MOSEI.zip is not downloadable anymore from the official site. Please contact '\
                            'the authors of the dataset to get access to the data.')
        out_unzip = self.data_processed_dir
        if os.path.exists(out_unzip):
            file_util.rmdir(out_unzip)
        dir_to_unzip = os.path.join(COMBINED_VIDEO_DIR, '*')
        unzip_command = f'unzip {external_file_path} {dir_to_unzip} -d {out_unzip}'
        print(f'Unzipping downloaded file {file_to_download}.')
        subprocess.run(unzip_command, shell=True)
        rename_dir_src = os.path.join(
            self.data_processed_dir, COMBINED_VIDEO_DIR)
        videos_original_dir = self._paths_for_data_key(C.DATA_KEY_VIDEOS_ORIGINAL)[0]
        rename_dir_target = videos_original_dir
        # Rename from Raw/Videos/Full/Combined to just videos
        os.rename(rename_dir_src, rename_dir_target)
        file_util.rmdir(os.path.join(
            self.data_processed_dir, 'Raw', 'Videos', 'Full'))

    def _get_dataset(self):   
        print('Obtaining external files for CMU MOSEI.')
        cmumosei_dataset = {}
        for data in DATA:
            data_dir = os.path.join(self.data_external_dir, f'cmumosei_{data}')
            recipe = DATA_SDK_KEYS[data]
            for external_file, file_size in EXTERNAL_FILES[data]:
                file_path = os.path.join(data_dir, external_file)
                if not os.path.exists(file_path) or os.stat(file_path).st_size != file_size:
                    if os.path.exists(file_path):
                        actual_size = os.stat(file_path).st_size
                        print(
                            f'File {file_path} does not have the expected file'
                            f' size {file_size} (has {actual_size}). '
                            ' Removing and will download again.')
                        # Wrong size so we remove it
                        os.remove(file_path)
                    else:
                        print(
                            f'File {file_path} does not exist. Will download again.')
                    # We just remove the corrupt file and keep the key in the
                    # recipe so that it is downloaded again
                    os.makedirs(data_dir, exist_ok=True)
                else:
                    del recipe[FILENAMES_TO_SDK_KEYS[external_file]]
            # This dataset is simply constructed so that the relevant files are
            # downloaded
            if len(recipe.keys()) > 0:
                mmdatasdk.mmdataset(recipe, data_dir)
            cmumosei_dataset[data] = mmdatasdk.mmdataset(data_dir)
        return cmumosei_dataset

    def _extract_data(self, corrupt_data_keys):
        cmumosei_dataset = self._get_dataset()
        train_videos = mmdatasdk.cmu_mosei.standard_folds.standard_train_fold
        val_videos = mmdatasdk.cmu_mosei.standard_folds.standard_valid_fold
        test_videos = mmdatasdk.cmu_mosei.standard_folds.standard_test_fold
        annotations = {}
        # We need a separate annotations file for the cropped faces because in some videos, no faces can be found.
        annotations_cropped = {}

        videos_original_dir = self._paths_for_data_key(C.DATA_KEY_VIDEOS_ORIGINAL)[0]
        video_file_paths = file_util.sorted_listdir(videos_original_dir)
        videos_dir = self._paths_for_data_key(C.DATA_KEY_VIDEOS)[0]
        video_intervals_dir = self._paths_for_data_key(C.DATA_KEY_VIDEO_INTERVALS)[0]

        audio_dir = self._paths_for_data_key(C.DATA_KEY_AUDIO)[0]
        spectrograms_dir = self._paths_for_data_key(C.DATA_KEY_AUDIO_SPECTROGRAMS)[0]
        landmarks_dir = self._paths_for_data_key(C.DATA_KEY_FACIAL_LANDMARKS)[0]
        embeddings_dir = self._paths_for_data_key(C.DATA_KEY_GLOVE_EMBEDDINGS)[0]
        texts_dir = self._paths_for_data_key(C.DATA_KEY_TEXTS)[0]

        # For some reason they don't get a matching equivalent in the features so we just skip them
        VIDEOS_TO_SKIP = ['9cxlcpbmrH0_6', '2cwNG0YuwtQ_6']

        cropping_faces = self.crop_face and\
            (C.DATA_KEY_VIDEOS in corrupt_data_keys or C.DATA_KEY_ANNOTATIONS in corrupt_data_keys)

        for video_idx, video_file_path in enumerate(video_file_paths):
            comp_seq = cmumosei_dataset['labels']['All Labels']
            original_video_filename = os.path.basename(video_file_path)
            original_video_filename_without_ext = os.path.splitext(original_video_filename)[0]
            print(f'Processing video {original_video_filename} ({video_idx + 1}/{len(video_file_paths)}).', flush=True)
            if original_video_filename_without_ext not in comp_seq.keys():
                print(f'No annotatios for video {video_file_path} found in the official dataset. Skipping.')
                continue
            video_path = os.path.join(videos_original_dir, video_file_path)
            comp_seq = comp_seq[original_video_filename_without_ext]
            intervals = comp_seq['intervals']
            labels = comp_seq['features']
            fold = None
            if original_video_filename_without_ext in train_videos:
                fold = 'train'
            if original_video_filename_without_ext in val_videos:
                fold = 'val'
            if original_video_filename_without_ext in test_videos:
                fold = 'test'
            if fold is None:
                print(
                    f'Did not find video {original_video_filename} in any of the offical folds. Skipping.')
                continue

            print(f'Found {len(intervals)} intervals for video.',
                  flush=True)
            for i, interval in enumerate(intervals):
                print(f'Processing interval {i + 1}.', end='\r')
                # Interval is in the format of e.g. 10.63 for 10 seconds and
                # 63 ms, so we convert it to ms by multiplying by 1000
                interval = interval * 1000
                # Entry key is video filename plus interval index
                identifier = f'{original_video_filename_without_ext}_{i}'
                if identifier in VIDEOS_TO_SKIP:
                    print(f'Skipping video {identifier} because it causes problems otherwise.')
                    continue
                video_filename = f'{identifier}.mp4'
                original_video_path = os.path.join(videos_original_dir, original_video_filename)
                # Dir for videos separated into intervals but in original size (will be used e.g. for cropping the faces)
                output_original_size_video_interval_file_path = os.path.join(video_intervals_dir, fold, video_filename)
                output_video_file_path = os.path.join(videos_dir, fold, video_filename)

                start = interval[0]
                end = interval[1]

                video_ok = None

                if C.DATA_KEY_VIDEO_INTERVALS in corrupt_data_keys:
                    # These are the full-sized videos but split into the intervals. They will be used e.g. for
                    # cropping the faces.
                    video_ok = video_util.cut_resize_video(
                        input_video_file_path=original_video_path,
                        output_video_file_path=output_original_size_video_interval_file_path,
                        with_sound=False,
                        start=start,
                        length=end - start)
                    if not video_ok:
                        continue

                if C.DATA_KEY_VIDEOS in corrupt_data_keys:
                    # We only need to resize the originally-sized videos after sectioning them into the intervals,
                    # if the user did not request to crop the faces. In the latter case, we will use the originally-
                    # sized videos to crop out the faces and then resize the videos.
                    if self.resize_scale and not self.crop_face:
                        video_ok = video_util.cut_resize_video(
                            input_video_file_path=original_video_path,
                            output_video_file_path=output_video_file_path,
                            with_sound=False,
                            size=self.resize_scale,
                            start=start,
                            length=end - start)
                        if not video_ok:
                            continue

                if cropping_faces:
                    # We don't put this into the main config because there is no sense in checking whether the number
                    # of files there is correct. The cropping process will always produce them.
                    videos_unmerged_dir = os.path.join(videos_dir + '_unmerged')
                    cropped_fold = os.path.join(videos_unmerged_dir, fold)
                    if not os.path.exists(cropped_fold):
                        os.makedirs(cropped_fold)
                    cropped_video_file_path = os.path.join(cropped_fold, video_filename)
                    print(f'Detecting faces in video {video_filename}. This may take a while.', flush=True)
                    cropped_number, cropped_coor = crop_face_util.crop_to_face_merge_video(
                        # Take the originally-sized video to detect the faces
                        input_video=output_original_size_video_interval_file_path,
                        split_video=cropped_video_file_path,
                        merged_video=output_video_file_path,
                        resize_scale=self.resize_scale,
                        batch_size=self.preprocessing_batch_size)
                    # In some rare cases, the automatic cropping does not find any faces in the video,
                    # this is the reason why we have to create a separate annotations file.
                    if cropped_number > 0:
                        video_ok = True
                    else:
                        video_ok = False
                        continue

                if video_ok is None:
                    # video_ok is None if the video data itself wasn't processed, because the number of video
                    # files is correct but not the audio, e.g. If this is the case, we check if the output video
                    # file exists. If it doesn't, we need to skip this sample. It might not exist because when cropping
                    # the video, there were no faces found so the output video file does not exist.
                    if not os.path.exists(output_video_file_path):
                        continue

                if not video_ok:
                    continue

                audio_file_path = os.path.join(audio_dir, fold, f'{identifier}.wav')
                if C.DATA_KEY_AUDIO in corrupt_data_keys:
                    audio_util.extract_audio_from_video_file(
                        video_path,
                        audio_file_path,
                        start=interval[0],
                        length=interval[1] - interval[0]
                    )
                # Spectrograms
                spectogram_file_path = os.path.join(spectrograms_dir, fold, f'{identifier}.npy' )
                if C.DATA_KEY_AUDIO_SPECTROGRAMS in corrupt_data_keys:
                    self._extract_audio_spectrogram(audio_file_path, spectogram_file_path)

                # Landmarks
                if C.DATA_KEY_FACIAL_LANDMARKS in corrupt_data_keys:
                    comp_seq_landmarks = cmumosei_dataset['highlevel']['OpenFace_2']
                    comp_seq_landmarks = comp_seq_landmarks[original_video_filename_without_ext]
                    features_landmarks = comp_seq_landmarks['features']
                    intervals_landmarks = comp_seq_landmarks['intervals']
                    landmarks_ind = np.where(((intervals_landmarks[:, 0] * 1000 >= interval[0]) & (
                        intervals_landmarks[:, 1] * 1000 <= interval[1])))
                    facial_landmarks = np.hstack((features_landmarks[landmarks_ind[0], 298:434],
                                                  (intervals_landmarks[landmarks_ind[0], :] - interval[0] / 1000)))
                    landmarks_file_path = os.path.join(landmarks_dir, fold, f'{identifier}.npy')
                    np.save(landmarks_file_path, facial_landmarks)

                # Embeddings
                embedding_ind = None
                if C.DATA_KEY_GLOVE_EMBEDDINGS in corrupt_data_keys:
                    comp_seq_embeddings = cmumosei_dataset['highlevel']['glove_vectors']
                    comp_seq_embeddings = comp_seq_embeddings[original_video_filename_without_ext]
                    features_embeddings = comp_seq_embeddings['features']
                    intervals_embedding = comp_seq_embeddings['intervals']
                    embedding_ind = np.where(((intervals_embedding[:, 0] * 1000 >= interval[0]) & (
                        intervals_embedding[:, 1] * 1000 <= interval[1])))
                    glove_embeddings = np.hstack((features_embeddings[embedding_ind[0], :],
                                                  (intervals_embedding[embedding_ind[0], :] - interval[0] / 1000)))
                    embeddings_file_path = os.path.join(embeddings_dir, fold, f'{identifier}.npy')
                    np.save(embeddings_file_path, glove_embeddings)

                # Texts
                text_file_path = os.path.join(texts_dir, fold, f'{identifier}.txt')
                if C.DATA_KEY_TEXTS in corrupt_data_keys:
                    if embedding_ind is None:
                        comp_seq_embeddings = cmumosei_dataset['highlevel']['glove_vectors']
                        comp_seq_embeddings = comp_seq_embeddings[original_video_filename_without_ext]
                        features_embeddings = comp_seq_embeddings['features']
                        intervals_embedding = comp_seq_embeddings['intervals']
                        embedding_ind = np.where(((intervals_embedding[:, 0] * 1000 >= interval[0]) & (
                            intervals_embedding[:, 1] * 1000 <= interval[1])))
                    words_seq = cmumosei_dataset['raw']['words'][original_video_filename_without_ext]['features']
                    texts = words_seq[embedding_ind[0]]
                    texts = texts.reshape(-1, ).astype('str')
                    texts = ' '.join(texts)
                    # MOSEI texts translated back from embeddings have this separator, we replace it by a whitespace
                    texts = texts.replace('sp', '')
                    with open(text_file_path, 'w') as texts_file:
                        texts_file.write(texts)

                if fold not in annotations:
                    annotations[fold] = {}
                if identifier not in annotations[fold]:
                    annotations[fold][identifier] = {}

                # str(entry_key) to force entry_key to be a string, otherwise
                # numbers like 991239_0 which indice the 0th part of video 991239
                # will be converted to 9912390 which is not what we want
                annotations[fold][identifier][C.BATCH_KEY_SENTIMENT] = float(labels[i][0])
                annotations[fold][identifier][C.BATCH_KEY_EMOTIONS] = labels[i][1:].tolist()
                annotations[fold][identifier]['video'] = identifier
                annotations[fold][identifier]['interval'] = i
                annotations[fold][identifier][C.BATCH_KEY_PERSON_ID] = original_video_filename_without_ext

        if C.DATA_KEY_ANNOTATIONS in corrupt_data_keys:
            annotations_file_path = self._paths_for_data_key(C.DATA_KEY_ANNOTATIONS)[0]
            with open(annotations_file_path, 'w+') as annotations_file:
                yaml.safe_dump(annotations, annotations_file)

    def _prepare_annotations_data(self, annotations: dict):
        new_annotations = {'train': {}, 'val': {}, 'test': {}}
        separator = '_'
        for subset in ['train', 'val', 'test']:
            subset_annotations = annotations[subset]
            if self.target_annotation == C.BATCH_KEY_EMOTIONS:
                if not self.multi_label and self.label_type == 'discrete':
                    for filename, annotation in subset_annotations.items():
                        key = filename
                        # Because we might have to iterate over the individual
                        # annotation and modify it
                        annotation = copy.deepcopy(annotation)
                        # We need an artificial new label for "neutral" for the entries
                        # that don't have any label set at all
                        emotions = np.zeros([7])
                        emotions[:6] = np.array(annotation['emotions'])
                        emotions[np.where(
                            emotions <= self.label_discretization_threshold)] = 0
                        emotions[np.where(
                            emotions > self.label_discretization_threshold)] = 1
                        emotions = emotions.astype(np.int32)
                        indices = np.where(emotions != 0)[0]
                        num_indices = indices.shape[0]
                        if num_indices > 1:
                            for i, idx in enumerate(indices):
                                annotation['emotions'] = idx
                                if i == 0:
                                    new_annotations[subset][f'{key}'] = annotation
                                else:
                                    new_annotations[subset][f'{key}{separator}{i}'] = annotation
                                    # We also store the filename in this entry because the
                                    # key itself is not only the filename anymore but appended with -i
                                    new_annotations[subset][f'{key}{separator}{i}']['filename'] = filename
                        if num_indices == 0:
                            annotation['emotions'] = 6
                elif self.multi_label and self.label_type == 'discrete':
                    for filename, annotation in subset_annotations.items():
                        length = len(filename)
                        # Get the "person" ID by using the ID of the video. We forgot to add the ID
                        # to the annotations file thats' why we add it here manually.
                        while length > 0:
                            if filename[length - 1] == separator:
                                break
                            length -= 1
                        annotation[C.BATCH_KEY_PERSON_ID] = filename[:length - 1]
                        key = filename
                        annotation['emotions_original'] = np.array(annotation['emotions'])
                        emotions = np.array(annotation['emotions'])
                        emotions[np.where(
                            emotions <= self.label_discretization_threshold)] = 0
                        emotions[np.where(
                            emotions > self.label_discretization_threshold)] = 1
                        if self.add_neutral_label and np.all(emotions == 0):
                            # All other emotions are 0, so we can reconstruct
                            # them with np.zeros and there's no problem
                            emotions = np.zeros([7])
                            emotions[6] = 1
                        # NOTE: np.float32 because for mulitlabel classification
                        # pytorch expects float labels
                        emotions = emotions.astype(np.float32)
                        annotation[C.BATCH_KEY_EMOTIONS] = emotions
                        annotation[C.BATCH_KEY_SENTIMENT] = int(annotation[C.BATCH_KEY_SENTIMENT])
                        new_annotations[subset][f'{key}'] = annotation
            elif self.target_annotation == C.BATCH_KEY_SENTIMENT:
                # Sentiment is from -1 to 1 but CUDA expects classes >= 0
                for subset in self.AVAILABLE_SUBSETS:
                    for key, value in annotations[subset].items():
                        if self.label_type == 'discrete':
                            # Discretize negative values into class 0, neutral values into class 1 and
                            # positive values into class 2
                            sentiment_value = int(value[C.BATCH_KEY_SENTIMENT])
                            if sentiment_value < 0:
                                sentiment_value = 0
                            elif sentiment_value == 0:
                                sentiment_value = 1
                            else:
                                sentiment_value = 2
                            new_annotations[subset][key][C.BATCH_KEY_SENTIMENT] = sentiment_value
                        else:
                            raise NotImplementedError(f'Continuous sentiment not implemented yet.')
                else:
                    raise NotImplementedError(f'Unknown target annotation {self.target_annotation}.')
        return new_annotations

    def _compute_class_weights(self, subset, annotations):
        print(f'Computing weights for {subset} classes...')
        if self.target_annotation == C.BATCH_KEY_EMOTIONS:
            # Hard-coded weights from Enrico
            self.class_weights[subset] = [1.0528500080108643, 5.215074062347412,
                                        5.3373003005981445, 19.48985481262207,
                                        7.0448384284973145, 23.688003540039062]
        else:
            collected_sentiment = np.arange(self.num_classes)
            for key, value in annotations[subset].items():
                sentiment = value[C.BATCH_KEY_SENTIMENT]
                collected_sentiment[sentiment] += 1
            N = float(np.sum(collected_sentiment))
            class_weights = N / collected_sentiment
            class_weights /= sum(class_weights)
            self.class_weights[subset] =  class_weights
