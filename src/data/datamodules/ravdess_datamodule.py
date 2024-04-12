from genericpath import isdir
import os
import shutil
import numpy as np
import wget
import glob
from zipfile import ZipFile

import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Subset

from data.datamodules import KFoldDataModule, VideoClassificationDataModule
from utils import constants as C, file_util, crop_face_util, audio_util, video_util


class RAVDESSDataModule(KFoldDataModule, VideoClassificationDataModule):

    DATASET_NAME = 'ravdess'

    AVAILABLE_SUBSETS = ['train', 'val']

    AVAILABLE_MODALITIES = [
        C.DATA_KEY_FRAMES_3D,
        C.DATA_KEY_FRAMES_2D,
        C.DATA_KEY_FRAMES_2D_3D,
        C.DATA_KEY_AUDIO_SPECTROGRAMS,
        C.DATA_KEY_TEXTS]

    VIDEO_FILES_TO_DOWNLOAD = {
        'Video_Speech_Actor_01.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_01.zip?download=1', 552968353),
        'Video_Speech_Actor_02.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_02.zip?download=1', 570673024),
        'Video_Speech_Actor_03.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_03.zip?download=1', 567824665),
        'Video_Speech_Actor_04.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_04.zip?download=1', 546560009),
        'Video_Speech_Actor_05.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_05.zip?download=1', 562966425),
        'Video_Speech_Actor_06.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_06.zip?download=1', 568517859),
        'Video_Speech_Actor_07.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_07.zip?download=1', 565652404),
        'Video_Speech_Actor_08.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_08.zip?download=1', 562573079),
        'Video_Speech_Actor_09.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_09.zip?download=1', 523489249),
        'Video_Speech_Actor_10.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_10.zip?download=1', 565580163),
        'Video_Speech_Actor_11.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_11.zip?download=1', 518634370),
        'Video_Speech_Actor_12.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_12.zip?download=1', 557928718),
        'Video_Speech_Actor_13.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_13.zip?download=1', 501877549),
        'Video_Speech_Actor_14.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_14.zip?download=1', 552484059),
        'Video_Speech_Actor_15.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_15.zip?download=1', 525205576),
        'Video_Speech_Actor_16.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_16.zip?download=1', 562416513),
        'Video_Speech_Actor_17.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_17.zip?download=1', 551688845),
        'Video_Speech_Actor_18.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_18.zip?download=1', 565655520),
        'Video_Speech_Actor_19.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_19.zip?download=1', 581121687),
        'Video_Speech_Actor_20.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_20.zip?download=1', 561349264),
        'Video_Speech_Actor_21.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_21.zip?download=1', 590292896),
        'Video_Speech_Actor_22.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_22.zip?download=1', 560907466),
        'Video_Speech_Actor_23.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_23.zip?download=1', 545177865),
        'Video_Speech_Actor_24.zip': ('https://zenodo.org/record/1188976/files/Video_Speech_Actor_24.zip?download=1', 595799165)
    }

    AUDIO_FILES_TO_DOWNLOAD = {
        'Audio_Speech_Actors_01-24.zip': ('https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1', 208468073)
    }

    def __init__(self, manual_folds: bool = None, remove_neutral: bool = False, **kwargs):
        kwargs['train_val_split'] = None
        super().__init__(**kwargs)
        self.data_external_dir = os.path.join(
            self.data_dir, C.EXTERNAL_DATA_DIR, C.DATASET_RAVDESS)
        self._SYMLINK_KEYS = [C.DATA_KEY_VIDEOS, C.DATA_KEY_AUDIO, C.DATA_KEY_AUDIO_SPECTROGRAMS, C.DATA_KEY_TEXTS]
        # We have to do this because we have symlinks from val to train for specific keys. If the user copied the dataset
        # to a new machine, the symlinks would be broken. Thus, we remove the symlinks and create them again.
        self._remove_symlinks()
        self._create_symlinks()
        self._annotations_per_person = None
        self._remove_neutral = remove_neutral
        assert not (manual_folds and self.num_folds), 'Cannot use manual folds and num_folds at the same time.'
        self.manual_folds = manual_folds
        if manual_folds:
            # The splits that Runfeng used
            self._manual_folds = [
                ([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], [23, 22], [0, 1, 2]),
                ([0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], [23, 22], [3, 4, 5]),
                ([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], [23, 22], [6, 7]),
                ([0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], [23, 22], [8, 9]),
                ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21], [23, 22], [12, 13, 14]),
                ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], [23, 22], [10, 11]),
                ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21], [23, 22], [15, 16]),
                ([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23], [0, 1], [17, 18]),
                ([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23], [0, 1], [19, 20]),
                ([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [0, 1], [23, 21, 22])]

    def _remove_symlinks(self):
        symlink_keys = self._SYMLINK_KEYS.copy()
        for i, symlink_key in enumerate(symlink_keys):
            if symlink_key in self._raw_data_key_for_data_key:
                symlink_keys[i] = self._raw_data_key_for_data_key[symlink_key]
        raw_data_paths = [self._paths_for_data_key(key)[0] for key in symlink_keys]
        for raw_data_path in raw_data_paths:
            val_raw_data_path = os.path.join(raw_data_path, 'val')
            # It's a symlink to train. The superclass doesn't like this, so
            # we remove it, let the superclass do it's thing, remove the created
            # val directory and re-create it again as a symlink. A bit hacky.
            if os.path.islink(val_raw_data_path):
                # Previously created symlink that we now remove
                os.remove(val_raw_data_path)
            elif os.path.exists(val_raw_data_path):
                # Maybe data extraction crashed before creating the actual symlinks (so only the path instantiation
                # of the superclass worked), thus we have a normal directory that we remove.
                shutil.rmtree(val_raw_data_path)

    def _create_symlinks(self):
        symlink_keys = self._SYMLINK_KEYS.copy()
        for i, symlink_key in enumerate(symlink_keys):
            if symlink_key in self._raw_data_key_for_data_key:
                symlink_keys[i] = self._raw_data_key_for_data_key[symlink_key]
        raw_data_paths = [self._paths_for_data_key(key)[0] for key in symlink_keys]
        for path in raw_data_paths:
            abs_path = os.path.abspath(path)
            val_path = os.path.join(abs_path, 'val')
            train_path = os.path.join(abs_path, 'train')
            if os.path.exists(train_path):
                os.symlink(train_path, val_path)

    def available_modalities(self):
        return self.AVAILABLE_MODALITIES

    def available_subsets(self):
        return self.AVAILABLE_SUBSETS

    def _raw_data_keys(self):
        return [C.DATA_KEY_VIDEOS_ORIGINAL, C.DATA_KEY_AUDIO]

    def _init_counts_for_data_keys(self):
        annotations_size = 115221
        # 5760 = 2880 * 2 since we have a symlink from val to train
        crop_varying_count = 1440 * 2

        # We also set the varying count for the audio spectrograms etc., since only the audio spectrograms that
        # match a video file are being processed. Since some videos don't have a cropped counterpart (since e.g.
        # no face was detected), the number of audio spectrograms is also smaller.
        # If we ever used the uncropped videos again, this would be a problem, because the number woulnd't be the
        # one we define in the dict here and the dataset would get reprocessed.
        # But since we will always just use the cropped version that's ok.
        update_dict = {
            C.DATA_KEY_VIDEOS_ORIGINAL: [1440],
            C.DATA_KEY_VIDEOS: [crop_varying_count],
            C.DATA_KEY_FRAMES_2D_3D: None,
            C.DATA_KEY_FRAMES_2D: [crop_varying_count],
            C.DATA_KEY_FRAMES_3D: [crop_varying_count],
            C.DATA_KEY_GLOVE_EMBEDDINGS: [crop_varying_count],
            C.DATA_KEY_AUDIO: [crop_varying_count],
            C.DATA_KEY_TEXTS: [crop_varying_count],
            C.DATA_KEY_AUDIO_SPECTROGRAMS: [crop_varying_count],
            C.DATA_KEY_ANNOTATIONS: [annotations_size],
        }

        for key, value in update_dict.items():
            self._data_configs[key].data_counts = value

    def _extract_raw_data(self, corrupt_data_configs):
        if not os.path.exists(self.data_external_dir):
            os.makedirs(self.data_external_dir)

        if C.DATA_KEY_VIDEOS_ORIGINAL in corrupt_data_configs:
            videos_original_dir = self._paths_for_data_key(C.DATA_KEY_VIDEOS_ORIGINAL)[0]
            self._download_extract_and_move_raw_data(self.VIDEO_FILES_TO_DOWNLOAD, videos_original_dir)

        if C.DATA_KEY_AUDIO in corrupt_data_configs:
            # audio_dir will be audio_cropped usually since we only process the cropped videos. The clean way to do it
            # would be to extract the audio into a 'audio' dir and then copy only the audio files to audio_cropped where
            # there were faces detected in the corresponding video. But since RAVDESS is so easy to crop and all videos
            # have detected faces, we just directly put the extracted audio files into audio_cropped.
            audio_dir = self._paths_for_data_key(C.DATA_KEY_AUDIO)[0]
            self._download_extract_and_move_raw_data(self.AUDIO_FILES_TO_DOWNLOAD, audio_dir)

    def _download_extract_and_move_raw_data(self, files_to_download, target_dir):
        target_dir = os.path.join(target_dir, 'train')
        # We now download and extract files. RAVDESS supplies data in zip files which contain
        # 1. The video files of one actor, e.g. Actor_01/03-01-01-01-01-01-01.mp4
        # 2. The audio files of all actors, e.g. Actor_01/03-01-01-01-01-01-01.wav, Actor_02/03-01-01-01-01-01-02.wav, etc.
        # We extract the zip files (24 individual files for the videos and one for the audios) which gives us
        # 24 directories in target_dir. Next, we move all the extracted (video and audio) files to the target_dir and
        # delete all the individual actor directories.
        for file_name, (url, size) in files_to_download.items():
            external_file = os.path.join(self.data_external_dir, file_name)
            need_to_download = False
            if not os.path.exists(external_file):
                print(f'File {file_name} does not exist, downloading from {url}')
                need_to_download = True
            elif os.path.getsize(external_file) != size:
                print(f'File {file_name} does not match expected file size {size}, downloading from {url}')
                os.remove(external_file)
                need_to_download = True
            if need_to_download:
                wget.download(url, out=self.data_external_dir)
            zip_file = ZipFile(external_file, 'r')
            print(f'Unpacking file {file_name}')
            zip_file.extractall(target_dir)
            zip_file.close()

        # There might be some other previously extracted files, thus we list only the directories in the target_dir.
        extracted_dirs = [dir for dir in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, dir))]
        for actor_dir in extracted_dirs:
            actor_dir_path = os.path.join(target_dir, actor_dir)
            files = os.listdir(actor_dir_path)
            for file in files:
                existing_file_path = os.path.join(target_dir, file)
                if os.path.exists(existing_file_path):
                    os.remove(existing_file_path)
                shutil.move(os.path.join(actor_dir_path, file), target_dir)
                # The audio files start with 03 for audio-only, but 01 for video files. We rename the audio files
                # to start with 01 so that they match the video files. Actually, 01 should be full audio-video but
                # somehow contains only video.
                filename_split = file.split('-')
                if filename_split[0] == '03':
                    filename_split[0] = '01'
                    renamed_file = '-'.join(filename_split)
                    renamed_file_path = os.path.join(target_dir, renamed_file)
                    os.rename(existing_file_path, renamed_file_path)
                if filename_split[0] == '02':
                    # The video files are supplied twice, once without sound and once with. Actually, all video files
                    # don't contain sound. Since they are there twice, we remove the ones without sound.
                    os.remove(existing_file_path)
            shutil.rmtree(actor_dir_path)

    def _extract_data(self, corrupt_data_keys):
        videos_original_dir = self._paths_for_data_key(C.DATA_KEY_VIDEOS_ORIGINAL)[0]

        videos_dir = self._paths_for_data_key(C.DATA_KEY_VIDEOS)[0]
        train_videos_dir = os.path.join(videos_dir, 'train')

        video_file_paths = glob.glob(os.path.join(videos_original_dir, 'train', '**', '*.mp4'), recursive=True)
        cropping_faces = self.crop_face and\
            (C.DATA_KEY_VIDEOS in corrupt_data_keys or\
                C.DATA_KEY_ANNOTATIONS in corrupt_data_keys or\
                    C.DATA_KEY_FACIAL_LANDMARKS in corrupt_data_keys)
        annotations_cropped = {'train': {}, 'val': {}, 'test': {}}
        annotations = {'train': {}, 'val': {}, 'test': {}}
        for idx, video_file_path in enumerate(video_file_paths):
            print(f'Processing video {video_file_path} ({idx + 1}/{len(video_file_paths)})', flush=True)
            video_filename = os.path.basename(video_file_path)
            identifier = file_util.get_filename_without_extension(video_filename)
            output_video_file_path = os.path.join(train_videos_dir, video_filename)
            video_ok = None
            split_identifier = identifier.split('-')
            emotion = int(split_identifier[2]) - 1

            if C.BATCH_KEY_VIDEOS in corrupt_data_keys and not self.crop_face:
                # We only need to do something if the videos are supposed to be resized, unpacking the dataset
                # happens everytime there is something wrong with it. This unpacking produces all the original
                # videos already that don't need any further processing. Above, we copy the videos from
                # videos_original to videos to make them available to the superclass.
                video_ok = video_util.cut_resize_video(
                    input_video_file_path=video_file_path,
                    output_video_file_path=output_video_file_path,
                    size=self.resize_scale,
                    with_sound=False
                )
                annotations['train'][identifier] = {
                    'emotions': emotion
                }
                annotations_cropped['val'][identifier] = {
                    'emotions': emotion
                }

            if cropping_faces:
                cropped_videos_unmerged_dir = self._paths_for_data_key(C.DATA_KEY_VIDEOS)[0] + '_unmerged'
                unmerged_fold_dir = os.path.join(cropped_videos_unmerged_dir, 'train')
                if not os.path.exists(unmerged_fold_dir):
                    os.makedirs(unmerged_fold_dir)
                unmerged_video_file_path = os.path.join(unmerged_fold_dir, video_filename)
                cropped_number, cropped_coor, landmarks = crop_face_util.crop_to_face_merge_video(
                    input_video=video_file_path,
                    split_video=unmerged_video_file_path,
                    merged_video=output_video_file_path,
                    batch_size=self.preprocessing_batch_size,
                    resize_scale=self.resize_scale,
                    with_landmarks=True)
                if cropped_number > 0:
                    video_ok = True
                    annotations_cropped['train'][identifier] = {
                        'emotions': emotion
                    }
                    annotations_cropped['val'][identifier] = {
                        'emotions': emotion
                    }
                    if C.DATA_KEY_FACIAL_LANDMARKS in corrupt_data_keys:
                        landmarks_file_path = os.path.join(
                            self._paths_for_data_key(C.DATA_KEY_FACIAL_LANDMARKS)[0], 'train', f'{identifier}.npy')
                        np.save(landmarks_file_path, landmarks)
                else:
                    video_ok = False

            if video_ok is None:
                if not os.path.exists(output_video_file_path):
                    continue
            elif not video_ok:
                continue

            audio_file_path = os.path.join(self._paths_for_data_key(C.DATA_KEY_AUDIO)[0], 'train', f'{identifier}.wav')

            if C.BATCH_KEY_AUDIO in corrupt_data_keys:
                # RAVDESS supplies audio files as zip files so we don't need to extract them here from the video files.
                # The audio gets downloaded and extracted above in _extract_raw_data.
                pass

            if C.BATCH_KEY_AUDIO_SPECTROGRAMS in corrupt_data_keys:
                spectrograms_dir = os.path.join(self._paths_for_data_key(C.DATA_KEY_AUDIO_SPECTROGRAMS)[0], 'train')
                spectogram_file_path = os.path.join(spectrograms_dir, f'{identifier}.npy')
                self._extract_audio_spectrogram(audio_file_path, spectogram_file_path)

            if C.BATCH_KEY_TEXTS in corrupt_data_keys:
                split_identifier = identifier.split('-')
                if split_identifier[4] == '01':
                    text = 'Kids are talking by the door'
                elif split_identifier[4] == '02':
                    text = 'Dogs are sitting by the door'
                texts_dir = os.path.join(self._paths_for_data_key(C.DATA_KEY_TEXTS)[0], 'train')
                text_file_path = os.path.join(texts_dir, f'{identifier}.txt')
                with open(text_file_path, 'w') as texts_file:
                    texts_file.write(text)

        if C.DATA_KEY_ANNOTATIONS in corrupt_data_keys:
            with open(self._paths_for_data_key(C.DATA_KEY_ANNOTATIONS)[0], 'w+') as annotations_file:
                print('cropping', cropping_faces)
                if cropping_faces:
                    yaml.safe_dump(annotations_cropped, annotations_file)
                else:
                    yaml.safe_dump(annotations, annotations_file)

    def _remove_corrupt_paths(self, corrupt_paths):
        # Remove the symlinks first, otherwise the superclass will crash because it will try to create the directories
        self._remove_symlinks()
        super()._remove_corrupt_paths(corrupt_paths)
        # The superclass will now have created the val directories again. We remove them
        # and create symlinks instead.
        self._remove_symlinks()
        self._create_symlinks()

    def _compute_class_weights(self, subset, annotations):
        weights = [1.875, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375]
        self._class_weights[subset] = weights

    def test_subset_name(self):
        return 'val'

    def _prepare_annotations_data(self, annotations):
        if self._annotations_per_person is None:
            self._annotations_per_person = {}
            # We only use the train annotations to create the folds, thus we only need the train annotations here.
            # The train and validation annotations don't differ, they are different by the data path to load them.
            for vid_id, annotation in annotations['train'].items():
                expression_id = vid_id.split('-')[2]
                if self._remove_neutral and expression_id == '01':
                    continue
                person_id = vid_id.split('-')[-1]
                sanitized_id = int(person_id) - 1
                if sanitized_id not in self._annotations_per_person:
                    self._annotations_per_person[sanitized_id] = {}
                annotation[C.BATCH_KEY_PERSON_ID] = sanitized_id
                self._annotations_per_person[sanitized_id][vid_id] = annotation
        annotations = {'train': {}, 'val': {}, 'test': {}}
        if self.manual_folds:
            final_folds = self._manual_folds
        elif self.num_folds is not None:
            persons = list(self._annotations_per_person.keys())
            num_samples = len(persons)
            # After super().setup we have only self.train_dataset since the dataset does not define val and test.
            # Since our code works with dictionaries that contain the targets, we fake the y dataset here.
            splits = self._skf.split(persons, np.zeros(num_samples))
            final_folds = []
            for val_train_split, test_split in splits:
                train_split, val_split = train_test_split(val_train_split, test_size=0.1, random_state=self.random_state)
                final_folds.append((train_split, val_split, test_split))
        else:
            final_folds = [[list(self._annotations_per_person.keys())] * 3]
            
        current_split = final_folds[self.current_fold]
        for train_id in current_split[0]:
            annotations['train'].update(self._annotations_per_person[train_id])
        for val_id in current_split[1]:
            annotations['val'].update(self._annotations_per_person[val_id])
        for test_id in current_split[2]:
            annotations['test'].update(self._annotations_per_person[test_id])
        if self.num_folds is not None:
            self.current_fold = (self.current_fold + 1) % self.num_folds
        return annotations
