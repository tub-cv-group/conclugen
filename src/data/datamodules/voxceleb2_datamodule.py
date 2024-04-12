import os
import glob
import shutil
import subprocess

from torch.utils.data import DataLoader
import torchaudio
import yaml
import natsort
from moviepy.editor import VideoFileClip

from . import VideoClassificationDataModule
from utils import file_util, dict_util, audio_util, list_util, video_util, transcripts_util, constants as C


EXTERNAL_VIDEO_FILES = {
    'vox2_dev_mp4_partaa': 32212254720,
    'vox2_dev_mp4_partab': 32212254720,
    'vox2_dev_mp4_partac': 32212254720,
    'vox2_dev_mp4_partad': 32212254720,
    'vox2_dev_mp4_partae': 32212254720,
    'vox2_dev_mp4_partaf': 32212254720,
    'vox2_dev_mp4_partag': 32212254720,
    'vox2_dev_mp4_partah': 32212254720,
    'vox2_dev_mp4_partai': 9257192977
}
EXTERNAL_AUDIO_FILES = {
    'vox2_dev_aaca': 10737418240,
    'vox2_dev_aacb': 10737418240,
    'vox2_dev_aacc': 10737418240,
    'vox2_dev_aacd': 10737418240,
    'vox2_dev_aace': 10737418240,
    'vox2_dev_aacf': 10737418240,
    'vox2_dev_aacg': 10737418240,
    'vox2_dev_aach': 2315355528
}


class VoxCeleb2DataModule(VideoClassificationDataModule):

    AVAILABLE_SUBSETS = ['train']

    AVAILABLE_MODALITIES = [
        C.DATA_KEY_FRAMES_3D,
        C.DATA_KEY_FRAMES_2D,
        C.DATA_KEY_FRAMES_2D_3D,
        C.DATA_KEY_AUDIO_SPECTROGRAMS,
        C.DATA_KEY_TEXTS]

    DATASET_NAME = C.DATASET_VOXCELEB2

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_external_dir = os.path.join(
            self.data_dir, C.EXTERNAL_DATA_DIR, C.DATASET_VOXCELEB2)

    def available_modalities(self):
        return self.AVAILABLE_MODALITIES

    def available_subsets(self):
        return self.AVAILABLE_SUBSETS

    def _raw_data_keys(self):
        return [C.DATA_KEY_VIDEOS_ORIGINAL]

    def _init_counts_for_data_keys(self):
        update_dict = {
            # More original videos because some are corrupt and we cannot resize them
            C.DATA_KEY_VIDEOS_ORIGINAL: [549984],
            C.DATA_KEY_VIDEOS: [534616],
            C.DATA_KEY_FRAMES_2D_3D: None,
            C.DATA_KEY_FRAMES_2D: [534616],
            C.DATA_KEY_FRAMES_3D: [534616],
            C.DATA_KEY_AUDIO: [534616],
            C.DATA_KEY_AUDIO_SPECTROGRAMS: [534616],
            C.DATA_KEY_TEXTS: [534616],
            C.DATA_KEY_ANNOTATIONS: [24057727]
        }
        for key, value in update_dict.items():
            self._data_configs[key].data_counts = value

    def _init_data_configs(self):
        super()._init_data_configs()

    def _verify_file_integrity(self, file_size_dict):
        for filename, size in file_size_dict.items():
            filepath = os.path.join(self.data_external_dir, filename)
            assert os.path.exists(filepath), f'You\'re missing the external file '\
                f'\'{filepath}\'. External files cannot be downloaded automatically '\
                'for this dataset. Please obtain it yourself.'
            assert os.stat(filepath).st_size == size, f'External file \'{filepath}\' '\
                f'has a wrong size of {os.stat(filepath).st_size} bytes (expected {size} bytes). '\
                f'Please download it again.'

    def _cat_and_unpack_files(self, file_size_dict, extracted_dir_name, extension, cat_zip_filename):
        cat_cmd = 'cat '
        for filename, size in file_size_dict:
            filepath = os.path.join(self.data_external_dir, filename)
            cat_cmd += filepath + ' '
        target_filepath = os.path.join(
            self.data_external_dir, cat_zip_filename)
        cat_cmd += '> ' + target_filepath
        subprocess.run(cat_cmd, shell=True)
        unzip_cmd = f'unzip {cat_zip_filename}'
        subprocess.run(unzip_cmd, shell=True)

        extracted_dir_path = os.path.join(
            self.data_external_dir, 'dev', extracted_dir_name)
        data = {}
        file_paths = glob.glob(os.path.join(
            extracted_dir_path, f'**/*.{extension}'), recursive=True)
        # Sorting is good for reproducibility
        file_paths = natsort.natsorted(file_paths)
        for file_path in file_paths:
            youtube_url_dir = os.path.split(file_path)[0]
            youtube_url = os.path.basename(youtube_url_dir)
            identity_dir = os.path.split(youtube_url_dir)[0]
            identity = os.path.basename(identity_dir)
            if identity not in data:
                data[identity] = {}
            if youtube_url not in data[identity]:
                data[identity][youtube_url] = []
            data[identity][youtube_url].append(file_path)
        return data

    def _extract_raw_data(self, corrupt_data_configs):
        # First check both because otherwise we don't need to unpack anything
        self._verify_file_integrity(EXTERNAL_VIDEO_FILES)
        self._verify_file_integrity(EXTERNAL_AUDIO_FILES)
        # We only support the train part of VoxCeleb2 since we only use it
        # for pretraining
        video_data = self._cat_and_unpack_files(
            EXTERNAL_VIDEO_FILES, 'mp4', 'mp4', 'vox2_dev_mp4.zip')
        audio_data = self._cat_and_unpack_files(
            EXTERNAL_AUDIO_FILES, 'wav', 'm4a', 'vox2_dev_aac.zip')
        # Get only the commmon entries, the dataset has some audio files that
        # have no corresponding video file and vice versa
        video_data, audio_data = dict_util.keep_common_nested_keys(
            [video_data, audio_data])
        video_target_dir = os.path.join(
            self.data_processed_dir, C.DIRNAME_VIDEOS, 'train')
        os.makedirs(video_target_dir, exist_ok=True)
        audio_target_dir = os.path.join(
            self.data_processed_dir, C.DIRNAME_AUDIO, 'train')
        os.makedirs(audio_target_dir, exist_ok=True)
        for identity, video_data in video_data.items():
            for youtube_url, video_file_paths in video_data.items():
                audio_file_paths = audio_data[identity][youtube_url]
                video_file_paths, audio_file_paths =\
                    list_util.intersect_file_paths_based_on_filename(
                        video_file_paths, audio_file_paths)
                for idx, video_file_path in enumerate(video_file_paths):
                    video_filename = os.path.basename(video_file_path)
                    video_target_filename = f'{identity}_{youtube_url}_{video_filename}'
                    video_target_file_path = os.path.join(
                        video_target_dir, video_target_filename)
                    os.rename(
                        video_file_path,
                        video_target_file_path)
                    audio_file_path = audio_data[identity][youtube_url][idx]
                    audio_filename = os.path.basename(audio_file_path)
                    audio_filename_without_extension = os.path.splitext(audio_filename)[
                        0]
                    audio_target_filename = f'{identity}_{youtube_url}_{audio_filename_without_extension}.wav'
                    audio_target_file_path = os.path.join(
                        audio_target_dir, audio_target_filename)
                    # Unfortunately, torchaudio cannot read m4a files, so we
                    # need to convert them to wav
                    convert_cmd = f'ffmpeg -i {audio_file_path} {audio_target_file_path}'
                    subprocess.run(convert_cmd, shell=True)

    def _extract_data(self, corrupt_data_keys):
        # Original videos only have one dir
        videos_original_dir = self._paths_for_data_key(C.DATA_KEY_VIDEOS_ORIGINAL)[0]
        video_filepaths = glob.glob(os.path.join(videos_original_dir, '**/*.mp4'), recursive=True)
            
        # Videos dir is only one dir
        videos_dir = self._paths_for_data_key(C.DATA_KEY_VIDEOS)[0]
        # Sorting is good for reproducibility
        video_filepaths = natsort.natsorted(video_filepaths)
        annotation_data = {}
        if C.DATA_KEY_VIDEOS in corrupt_data_keys and not C.DATA_KEY_ANNOTATIONS in corrupt_data_keys:
            # If the annotations file is corrupt we need to recreate it. But we cannot just create an entry in the dict
            # for each original video because some cannot be resized properly for whatever reason. If the number of
            # processed videos is already ok (i.e. the files on disk match the expected number), the resizing will be
            # skipped in the coming for loop and we won't know whether the resizing would have worked. To overcome this,
            # we simply list all files in the processed videos directory and use that as the basis for the annotations.
            # If the number of processed video files is not correct (i.e. prepraation_config[C.NUM_VIDEOS_OK] is False),
            # we don't have a problem because the video files will be resized again and we will know which ones have failed.
            processed_video_files = file_util.sorted_listdir(os.path.join(videos_dir, 'train'))
            # Keep the filename without extension to obtain identifier
            processed_video_files = [os.path.splitext(os.path.basename(f))[0] for f in processed_video_files]
        if self.preprocessing_batch_size is not None:
            accumulated_transcripts = []
        audio_dir = self._paths_for_data_key(C.DATA_KEY_AUDIO)[0]
        audio_extension = self._extension_for_data_key(C.DATA_KEY_AUDIO)[0]
        spectrograms_dir = self._paths_for_data_key(C.DATA_KEY_AUDIO_SPECTROGRAMS)[0]
        text_dir = self._paths_for_data_key(C.DATA_KEY_TEXTS)[0]
        for idx, video_filepath in enumerate(video_filepaths):
            print(
                f'Processing video {idx + 1}/{len(video_filepaths)}', end='\r')
            video_filename = os.path.basename(video_filepath)
            identifier = video_filename.replace('.mp4', '')
            # Some videos appear to be broken, we then also have to skip
            # spectrograms and texts
            video_data_ok = False
            target_path = os.path.join(videos_dir, 'train', f'{identifier}.mp4')
            if C.DATA_KEY_VIDEOS in corrupt_data_keys:
                if self.resize_scale is not None:
                    # Function returns false if there was an error processing
                    # the video
                    video_data_ok = video_util.cut_resize_video(
                        input_video_file_path=video_filepath,
                        output_video_file_path=target_path,
                        with_sound=False,
                        size=self.resize_scale)
                else:
                    shutil.copyfile(video_filepath, target_path)
                    video_data_ok = True
            else:
                # If the videos are not corrupt, we need to tell subsequent code if the video is ok, i.e. it has
                # been copied to the target path and exists.
                video_data_ok = os.path.exists(target_path)
            audio_wave_file = os.path.join(audio_dir, 'train', f'{identifier}.{audio_extension}')
            if C.DATA_KEY_AUDIO_SPECTROGRAMS in corrupt_data_keys and video_data_ok:
                spectrogram_file_path = os.path.join(spectrograms_dir, 'train', f'{identifier}.npy')
                self._extract_audio_spectrogram(audio_wave_file, spectrogram_file_path)
            process_text = C.DATA_KEY_TEXTS in corrupt_data_keys
            if process_text and video_data_ok:
                text_file_path = os.path.join(text_dir, 'train', f'{identifier}.txt')
                if self.preprocessing_batch_size is not None:
                    accumulated_transcripts.append(
                        [audio_wave_file, text_file_path])
                else:
                    transcripts_util.extract_transcripts_from_wave_files(
                        audio_wave_file, text_file_path)
            if C.DATA_KEY_ANNOTATIONS in corrupt_data_keys:
                if C.DATA_KEY_VIDEOS not in corrupt_data_keys:
                    # This is the case we have mentioned above when setting processed_video_files. The annotations
                    # file is corrupt but the number of processed videos is ok. We therefore use the list of processed
                    # videos as the basis for the annotations, i.e. check if the list contains the current identifier.
                    if identifier in processed_video_files:
                        annotation_data[identifier] = {
                            'emotions': 0
                        }
                # If the number of videos is not ok, we need to check whether resizing the video has successfully worked
                elif video_data_ok:
                    annotation_data[identifier] = {
                        # Will not be used since we are only pretraining on VoxCeleb2
                        # but we need an entry for the identifier
                        'emotions': 0
                    }
            if process_text:
                batch_processing = self.preprocessing_batch_size is not None and\
                                    idx % self.preprocessing_batch_size == 0 and\
                                    idx > 0
                if batch_processing or self.preprocessing_batch_size is None:
                    audio_wave_files, text_files = zip(*accumulated_transcripts)
                    audio_wave_files = list(audio_wave_files)
                    text_files = list(text_files)
                    transcripts_util.extract_transcripts_from_wave_files(
                        audio_wave_files, text_files)
                    accumulated_transcripts = []

        if C.DATA_KEY_ANNOTATIONS in corrupt_data_keys:
            annotations_file_path = self._paths_for_data_key(C.DATA_KEY_ANNOTATIONS)[0]
            with open(annotations_file_path, 'w+') as f:
                annotations = {
                    'train': annotation_data
                }
                yaml.dump(annotations, f)

    def _compute_class_weights(self, subset, annotations):
        raise NotImplementedError(
            'VoxCeleb2 should not be used for classification '
            'but only for pretraining')

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError(
            'Validation is not implemented for VoxCeleb2DataModule')

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError(
            'Test is not implemented for VoxCeleb2DataModule')
