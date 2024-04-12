import glob
import os
import shutil
import subprocess
from typing import List
import sys

import wget
import cv2
import numpy as np
import yaml
from moviepy.editor import *
import torchaudio

from data.datamodules import DataConfig, VideoClassificationDataModule
from utils import constants as C
from utils import file_util, audio_util, video_util, crop_face_util, transcripts_util
from utils import label_util

FILES_TO_DOWNLOAD = {
    'CAER.zip': 'https://drive.google.com/uc?export=download&id=1JsdbBkulkIOqrchyDnML2GEmuwi6E_d2&confirm=t&uuid=62eac424-9f39-42d6-935a-866de917f481'
}

FILES_TO_DOWNLOAD_SIZES = {
    'CAER.zip': 7634321773
}


class CAERVideoBaseDataModule(VideoClassificationDataModule):

    DATASET_NAME = C.DATASET_CAER

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_external_dir = os.path.join(
            self.data_dir, C.EXTERNAL_DATA_DIR, C.DATASET_CAER)
        self.data_external_videos_dir = os.path.join(
            self.data_external_dir, 'RAW', 'Videos', 'Full', 'Combined')

    def _extract_raw_data(self, corrupt_data_keys):
        file_to_download = 'CAER.zip'
        size = FILES_TO_DOWNLOAD_SIZES[file_to_download]
        external_file_path = os.path.join(
            self.data_external_dir, file_to_download)
        if not os.path.exists(external_file_path) or os.stat(external_file_path).st_size != size:
            print(f'Downloading missing external file {file_to_download}.')
            print('f: ', FILES_TO_DOWNLOAD[file_to_download], "out: ",self.data_external_dir )
             # Specify the desired output filename
            desired_filename = 'CAER.zip'  # Change this to your desired filename

            # Define the full path where the downloaded file should be saved, including the filename
            destination_path = os.path.join(self.data_external_dir, desired_filename)
            os.makedirs(self.data_external_dir, exist_ok=True)
            #Download the file with the specified output filename
            wget.download(
                url=FILES_TO_DOWNLOAD[file_to_download],
                out=destination_path)
        out_unzip = self.data_processed_dir
        if os.path.exists(out_unzip):
            file_util.rmdir(out_unzip)
        external_file_path = os.path.join(external_file_path)
        unzip_command = f'unzip {external_file_path} -d {out_unzip}'
        subprocess.run(unzip_command, shell=True)

        rename_dir_src = os.path.join(self.data_processed_dir, 'CAER')
        rename_dir_target = self._paths_for_data_key(
            C.BATCH_KEY_VIDEOS_ORIGINAL)[0]
        # Rename from Raw/Videos/Full/Combined to just videos
        os.rename(rename_dir_src, rename_dir_target)
        # rename validation to val
        val_dir = os.path.join(rename_dir_target, 'validation')
        val_target_dir = os.path.join(rename_dir_target, 'val')
        os.rename(val_dir, val_target_dir)
        print(f'Unzipping downloaded file {file_to_download}.')


class CAERVideoDataModule(CAERVideoBaseDataModule):

    AVAILABLE_SUBSETS = ['train', 'val', 'test']

    AVAILABLE_MODALITIES = [
        C.BATCH_KEY_FRAMES_2D_3D,
        C.BATCH_KEY_FRAMES_2D,
        C.BATCH_KEY_FRAMES_3D,
        C.BATCH_KEY_AUDIO_SPECTROGRAMS,
        C.BATCH_KEY_GLOVE_EMBEDDINGS,
        C.BATCH_KEY_TEXTS]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def available_modalities(self):
        return self.AVAILABLE_MODALITIES

    def available_subsets(self):
        return ['train', 'val', 'test']

    def _raw_data_keys(self):
        return [C.DATA_KEY_VIDEOS_ORIGINAL]

    def _init_data_configs(self):
        self._data_configs[C.DATA_KEY_VIDEOS_ORIGINAL].extensions = ['avi']
        super()._init_data_configs()

    def _init_counts_for_data_keys(self):
        update_dict = {
            C.DATA_KEY_ANNOTATIONS: [407087],
            C.DATA_KEY_VIDEOS_ORIGINAL: [13175],
            C.DATA_KEY_FRAMES_2D_3D: None,
            C.DATA_KEY_VIDEOS: [13175],
            C.DATA_KEY_FRAMES_2D: [13175],
            C.DATA_KEY_FRAMES_3D: [13175],
            C.DATA_KEY_AUDIO: [13175],
            C.DATA_KEY_TEXTS: [13175],
            C.DATA_KEY_AUDIO_SPECTROGRAMS: [13175]
        }
        for key, value in update_dict.items():
            self._data_configs[key].data_counts = value

    def _extract_data(self, corrupt_data_keys):
        annotations = {'train': {}, 'val': {}, 'test': {}}
        annotations_cropped = {'train': {}, 'val': {}, 'test': {}}
        videos_original_dir = self._paths_for_data_key(
            C.DATA_KEY_VIDEOS_ORIGINAL)[0]
        videos_dir = self._paths_for_data_key(C.DATA_KEY_VIDEOS)[0]
        cropping_faces = self.crop_face and (C.DATA_KEY_VIDEOS in corrupt_data_keys)
        for subset in self.available_subsets():
            original_videos_subset_dir = os.path.join(
                videos_original_dir, subset)
            videos_subset_dir = os.path.join(videos_dir, subset)
            video_file_paths = glob.glob(os.path.join(original_videos_subset_dir, '**', '*.avi'), recursive=True)
            for original_video_file_path in video_file_paths:
                # CAER is organized in a way that directories are the class labels
                parent_dir = file_util.get_parent_dir(original_video_file_path)
                identifier = file_util.get_filename_without_extension(original_video_file_path)
                output_filename = f'{parent_dir}_{identifier}.mp4'
                file = os.path.basename(original_video_file_path)
                target_video_file_path = os.path.join(videos_subset_dir, output_filename)
                print(f'Processing video {original_video_file_path}.', flush=True)

                if C.DATA_KEY_VIDEOS in corrupt_data_keys and not self.crop_face:
                    # We only need to do something if the videos are supposed to be resized, unpacking the dataset
                    # happens everytime there is something wrong with it. This unpacking produces all the original
                    # videos already that don't need any further processing. Above, we copy the videos from
                    # videos_original to videos to make them available to the superclass.
                    video_ok = video_util.cut_resize_video(
                        input_video_file_path=original_video_file_path,
                        output_video_file_path=target_video_file_path,
                        size=self.resize_scale,
                        with_sound=False
                    )
                else:
                    # Same for cropping and resizing, so we just check this one path here
                    video_ok = os.path.exists(target_video_file_path)
                # NOTE: if not cropping_faces because then the resized videos are not of our concern. The faces
                # cropping uses the converted originally-sized videos.
                if cropping_faces: #self crop face UND (defekte videos OR defekte annotations)
                    unmerged_dir = os.path.join(videos_dir + '_unmerged', subset)
                    os.makedirs(unmerged_dir, exist_ok=True)
                    split_video_path = os.path.join(unmerged_dir, output_filename)
                    merged_video_path = os.path.join(videos_subset_dir, output_filename)
                    cropped_number, cropped_coor, landmarks = crop_face_util.crop_to_face_merge_video(
                        input_video=original_video_file_path,
                        split_video=split_video_path,
                        merged_video=merged_video_path,
                        resize_scale=self.resize_scale,
                        batch_size=self.preprocessing_batch_size,
                        with_landmarks=True)
                    if cropped_number > 0:
                        video_ok = True
                        if C.DATA_KEY_FACIAL_LANDMARKS in corrupt_data_keys:
                            landmarks_file_path = os.path.join(
                                self._paths_for_data_key(C.DATA_KEY_FACIAL_LANDMARKS)[0], subset, f'{identifier}.npy')
                            np.save(landmarks_file_path, landmarks)
                    else:
                        video_ok = False

                if not video_ok:
                    print(f'Skipping corrupt video {original_video_file_path}. This original video will not be used.')
                    continue
                else:
                    annotations_cropped[subset][f'{parent_dir}_{identifier}'] = {
                        'emotions': label_util.class_index_for_fer_label(parent_dir)
                    }

                audio_dir = self._paths_for_data_key(C.DATA_KEY_AUDIO)[0]
                audio_file_path = os.path.join(audio_dir, subset, f'{parent_dir}_{identifier}' + '.wav')

                if C.DATA_KEY_AUDIO in corrupt_data_keys:
                    audio_util.extract_audio_from_video_file(
                        original_video_file_path,
                        audio_file_path)

                if C.DATA_KEY_AUDIO_SPECTROGRAMS in corrupt_data_keys:
                    spectrograms_dir = os.path.join(self._paths_for_data_key(C.DATA_KEY_AUDIO_SPECTROGRAMS)[0], subset)
                    spectogram_file_path = os.path.join(spectrograms_dir, f'{parent_dir}_{identifier}' + '.npy')
                    video_clip = VideoFileClip(original_video_file_path)
                    video_length = video_clip.reader.nframes
                    wav_file = torchaudio.load(audio_file_path)[0]
                    self._extract_audio_spectrogram(audio_file_path, spectogram_file_path)

                if C.DATA_KEY_TEXTS in corrupt_data_keys:
                    texts_file_path = os.path.join(self._paths_for_data_key(C.DATA_KEY_TEXTS)[0], subset, f'{parent_dir}_{identifier}' + '.txt')
                    transcripts_util.extract_transcripts_from_wave_files(audio_file_path, texts_file_path)

        if C.DATA_KEY_ANNOTATIONS in corrupt_data_keys:
            annotations_file_path = self._paths_for_data_key(
                C.DATA_KEY_ANNOTATIONS)[0]
            with open(annotations_file_path, 'w+') as annotations_file:
                yaml.safe_dump(annotations_cropped, annotations_file)

    def _compute_class_weights(self, subset, annotations):
        print(f'Computing weights for {subset} classes...')
        collected_emotions = []
        class_weights = np.zeros([7])
        for key, value in annotations[subset].items():
            emotions = np.array(value['emotions'])
            collected_emotions.append(emotions)
        collected_emotions = np.array(collected_emotions)
        for i in range(0, 7):
            class_weights[i] = np.count_nonzero(collected_emotions == i)
        class_weights /= sum(class_weights)
        self.class_weights[subset] = class_weights
