import glob
import os
import subprocess
import collections
import random
import csv
import shutil

import wget
import numpy as np
from moviepy.editor import *
import torchaudio
import yaml
import torchtext

from data.datamodules import VideoClassificationDataModule
from utils import constants as C
from utils import file_util, audio_util, video_util
from utils import crop_face_util

FILES_TO_DOWNLOAD = {
    'meld.tar.gz': 'https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz'
}

FILES_TO_DOWNLOAD_SIZES = {
    'meld.tar.gz': 10878146150
}

SUB_DIRS_TO_UNZIP = {
    'test.tar.gz': ['output_repeated_splits_test', 'test'],
    'dev.tar.gz': ['dev_splits_complete', 'val'],
    'train.tar.gz': ['train_splits', 'train']
}


class MELDDataModule(VideoClassificationDataModule):

    DATASET_NAME = C.DATASET_MELD

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_external_dir = os.path.join(
            self.data_dir, C.EXTERNAL_DATA_DIR, C.DATASET_MELD)

    def _extract_raw_data(self, corrupt_data_configs):
        file_to_download = 'meld.tar.gz'
        size = FILES_TO_DOWNLOAD_SIZES[file_to_download]
        if not os.path.exists(self.data_external_dir):
            os.makedirs(self.data_external_dir)
        external_file_path = os.path.join(
            self.data_external_dir, file_to_download)
        external_unzipped_dir = os.path.join(
            self.data_external_dir, C.DATASET_MELD)
        if os.path.exists(external_unzipped_dir):
            file_util.rmdir(external_unzipped_dir)
        if not os.path.exists(external_file_path) or os.stat(external_file_path).st_size != size:
            print(f'Downloading missing external file {file_to_download}.')
            wget.download(
                url=FILES_TO_DOWNLOAD[file_to_download],
                out=external_file_path)
        unzip_command = f'tar -zxvf {external_file_path} -C {self.data_external_dir}'
        subprocess.run(unzip_command, shell=True)
        os.rename(os.path.join(self.data_external_dir, 'MELD.Raw'), external_unzipped_dir)

        out_unzip = self.data_processed_dir
        # All data configs because one of them might not be corrupted (could actually only be the videos,
        # not the original videos)
        videos_original_dir = self._paths_for_data_key(C.DATA_KEY_VIDEOS_ORIGINAL)[0]
        videos_dir = self._paths_for_data_key(C.DATA_KEY_VIDEOS)[0]

        print('Unzipping external files...')
        for sub_dir, unzipped_dir in SUB_DIRS_TO_UNZIP.items():
            unzip_command = f'tar -zxvf {external_unzipped_dir}/{sub_dir} -C {videos_original_dir}'
            subprocess.run(unzip_command, shell=True)
            file_videos = os.listdir(os.path.join(
                videos_original_dir, unzipped_dir[0]))
            for file_video in file_videos:
                if file_video.split('/')[-1][0] == '.':
                    os.remove(os.path.join(videos_original_dir, unzipped_dir[0], file_video))
            # The videos are extracted into something like train/train_splits, so we need to move them to train
            source_videos_dir = os.path.join(videos_original_dir, unzipped_dir[0])
            target_video_dir = os.path.join(videos_original_dir, unzipped_dir[1])
            for file_video in file_util.sorted_listdir(source_videos_dir, '.mp4'):
                target_video_file_path = os.path.join(target_video_dir, file_video)
                if os.path.exists(target_video_file_path):
                    os.remove(target_video_file_path)
                shutil.move(os.path.join(source_videos_dir, file_video), target_video_dir)
            file_util.rmdir(source_videos_dir)
        mv_command = f'mv {external_unzipped_dir}/dev_sent_emo.csv {out_unzip}/val.csv'
        subprocess.run(mv_command, shell=True)
        mv_command = f'mv {external_unzipped_dir}/test_sent_emo.csv {out_unzip}/test.csv'
        subprocess.run(mv_command, shell=True)
        mv_command = f'mv {videos_original_dir}/train_sent_emo.csv {out_unzip}/train.csv'
        subprocess.run(mv_command, shell=True)
        file_util.rmdir(external_unzipped_dir)
        old_splits_dir = os.path.join(videos_dir, '._output_repeated_splits_test')
        if os.path.exists(old_splits_dir):
            file_util.rmdir(old_splits_dir)

        # delete all files with final_videos_test in test subset since no subtitles can be found in csv
        videos_without_subtitles = glob.glob(os.path.join(videos_original_dir, 'test', '*final_videos*'))
        for file in videos_without_subtitles:
            if 'final_videos' in file:
                os.remove(file)
                print(f'Removing {file} which does not have subtitles in the original dataset.')


class MELDVideoDataModule(MELDDataModule):

    AVAILABLE_SUBSETS = ['train', 'val', 'test']

    AVAILABLE_MODALITIES = [
        C.BATCH_KEY_FRAMES_2D_3D,
        C.BATCH_KEY_FRAMES_2D,
        C.BATCH_KEY_FRAMES_3D,
        C.BATCH_KEY_AUDIO_SPECTROGRAMS,
        C.BATCH_KEY_TEXTS,
        C.BATCH_KEY_GLOVE_EMBEDDINGS]

    def __init__(
        self,
        **kwargs
    ):
        """Init function of MELDVideoDataModule. The default values for the
        number of frames come from Soujanya's MELD paper:
        https://arxiv.org/pdf/1810.02508.pdf

        Args:
            num_frames_to_consider_for_extraction (int): The number of frames to
            consider in each video for frame extraction. E.g. 50 frames could be
            considered of a video with 150 frames, and from these 50 frames we
            could extract 8 evenly spaced frames for a sequence. Defaults to 50.
            num_frames_to_extract_for_sequence (int): The number of frames to
            evenly extract from the number of frames to consider. Defaults to 8.
        """
        super().__init__(**kwargs)

    def available_modalities(self):
        return self.AVAILABLE_MODALITIES

    def available_subsets(self):
        return self.AVAILABLE_SUBSETS

    def _raw_data_keys(self):
        return [C.DATA_KEY_VIDEOS_ORIGINAL]

    def _init_counts_for_data_keys(self):
        annotations_size = 717333
        crop_varying_count = 13698
        if self.crop_face:
            crop_varying_count = 13302
            annotations_size = 933322

        # We also set the varying count for the audio spectrograms etc., since only the audio spectrograms that
        # match a video file are being processed. Since some videos don't have a cropped counterpart (since e.g.
        # no face was detected), the number of audio spectrograms is also smaller.
        # If we ever used the uncropped videos again, this would be a problem, because the number woulnd't be the
        # one we define in the dict here and the dataset would get reprocessed.
        # But since we will always just use the cropped version that's ok.
        update_dict = {
            C.DATA_KEY_VIDEOS_ORIGINAL: [13715],
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

    def _init_data_configs(self):
        self._data_configs[C.DATA_KEY_ANNOTATIONS].dependencies = [C.DATA_KEY_VIDEOS_ORIGINAL, C.DATA_KEY_VIDEOS]
        super()._init_data_configs()

    def _extract_data(self, corrupt_data_keys):
        if C.DATA_KEY_GLOVE_EMBEDDINGS in corrupt_data_keys:
            glove = torchtext.vocab.GloVe(name='840B', dim=300)
            tokens_split = torchtext.data.get_tokenizer('basic_english')

        # Videos which exist but don't have an annotation
        corrupted_videos = {
            'train': ['dia110_utt7', 'dia125_utt3'],
            'val': ['dia110_utt7'],
            'test': []
        }
        annotations = {'train': {}, 'test': {}, 'val': {}}
        # To clear the print line
        len_last_print = 0

        for fold in ['train', 'val', 'test']:
            # Load the subtitles
            csv_file_path = os.path.join(self.data_processed_dir, f'{fold}.csv')
            all_subtitles = {}
            with open(csv_file_path, 'r') as csv_file:
                csv_reader = csv.DictReader(
                    csv_file, delimiter=',', quotechar='\"')
                for i, line in enumerate(csv_reader):
                    # Replacing \x92s because that's an apostrophe and it's wrongly encoded
                    diag_id = line['Dialogue_ID']
                    utt_id = line['Utterance_ID']
                    if diag_id not in all_subtitles:
                        all_subtitles[diag_id] = {}
                    all_subtitles[diag_id][utt_id] = line['Utterance'].replace(
                        '\\x92', '\'')
                    v_name = f'dia{diag_id}_utt{utt_id}'
                    emotion = self.labels.index(line['Emotion'])
                    sentiment = C.MELD_SENTIMENT_LABELS[line['Sentiment']]
                    annotations[fold][v_name] = {
                        C.BATCH_KEY_EMOTIONS: emotion,
                        C.BATCH_KEY_SENTIMENT: sentiment,
                        C.BATCH_KEY_PERSON_ID: line['Speaker']}

            # We need to remove the corrupted videos from the annotations
            for corrupted_video in corrupted_videos[fold]:
                if corrupted_video in annotations[fold]:
                    del annotations[fold][corrupted_video]

            # We know that there is only one original videos dir
            videos_original_dir = os.path.join(self._paths_for_data_key(C.DATA_KEY_VIDEOS_ORIGINAL)[0], fold)
            videos_dir = os.path.join(self._paths_for_data_key(C.DATA_KEY_VIDEOS)[0], fold)
            original_video_filepaths = file_util.sorted_listdir(videos_original_dir)
            # We need to process the cropped videos if self.crop_face is true together with if the number of videos is
            # not ok or if the annotations file size is not ok
            cropping_faces = self.crop_face and (C.DATA_KEY_VIDEOS in corrupt_data_keys)

            for video_idx, video_filepath in enumerate(original_video_filepaths):
                info = f'Processing video {video_filepath} ({video_idx + 1}/{len(original_video_filepaths)}).'
                print(info, flush=True)
                # Original is just the video we will use, we don't have intervals like in MOSEI for example
                original_video_file_path = os.path.join(videos_original_dir, video_filepath)
                identifier = file_util.get_filename_without_extension(video_filepath)
                video_filename = f'{identifier}.mp4'

                if identifier in corrupted_videos and fold == 'train':
                    os.remove(original_video_file_path)
                    del annotations[fold][identifier]
                    continue

                split_video_name = identifier.split('_')
                # part of the videos in test are named with final_videos_testdia...
                if 'dia' not in split_video_name[0] or 'utt' not in split_video_name[1]:
                    print(
                        f'Skipping wrong annotation data provided by dataset for video {identifier}.')
                    if identifier in annotations[fold]:
                        del annotations[fold][identifier]
                    continue
                diag_id = split_video_name[0].split('dia')[1]
                utt_id = split_video_name[1].split('utt')[1]
                if diag_id not in all_subtitles or utt_id not in all_subtitles[diag_id]:
                    print(f'Skipping video {video_filepath} since no subtitles could be found '
                          f'in the CSV file {csv_file_path} provided by the dataset.')
                    if identifier in annotations[fold]:
                        del annotations[fold][identifier]
                    continue

                output_video_file_path = os.path.join(videos_dir, video_filename)

                if C.DATA_KEY_VIDEOS in corrupt_data_keys and not self.crop_face:
                    # We only need to do something if the videos are supposed to be resized, unpacking the dataset
                    # happens everytime there is something wrong with it. This unpacking produces all the original
                    # videos already that don't need any further processing. Above, we copy the videos from
                    # videos_original to videos to make them available to the superclass.
                    video_ok = video_util.cut_resize_video(
                        input_video_file_path=original_video_file_path,
                        output_video_file_path=output_video_file_path,
                        size=self.resize_scale,
                        with_sound=False
                    )
                else:
                    # Same for cropping and resizing, so we just check this one path here
                    video_ok = os.path.exists(output_video_file_path)

                if cropping_faces:
                    cropped_videos_unmerged_dir = self._paths_for_data_key(C.DATA_KEY_VIDEOS)[0] + '_unmerged'
                    unmerged_fold_dir = os.path.join(cropped_videos_unmerged_dir, fold)
                    if not os.path.exists(unmerged_fold_dir):
                        os.makedirs(unmerged_fold_dir)
                    unmerged_video_file_path = os.path.join(unmerged_fold_dir, video_filename)
                    cropped_number, cropped_coor, landmarks = crop_face_util.crop_to_face_merge_video(
                        input_video=original_video_file_path,
                        split_video=unmerged_video_file_path,
                        merged_video=output_video_file_path,
                        resize_scale=self.resize_scale,
                        batch_size=self.preprocessing_batch_size,
                        with_landmarks=True)
                    if cropped_number > 0:
                        video_ok = True
                        if C.DATA_KEY_FACIAL_LANDMARKS in corrupt_data_keys:
                            landmarks_file_path = os.path.join(
                                self._paths_for_data_key(C.DATA_KEY_FACIAL_LANDMARKS)[0], fold, f'{identifier}.npy')
                            np.save(landmarks_file_path, landmarks)
                    else:
                        video_ok = False

                if not video_ok:
                    del annotations[fold][identifier]
                    print(f'Skipping video {original_video_file_path}. This original video will not be used '\
                          '(either too few dected faces or corrupt in original dataset).')
                    continue

                audio_file_path = os.path.join(self._paths_for_data_key(C.DATA_KEY_AUDIO)[0], fold, f'{identifier}.wav')

                if C.DATA_KEY_AUDIO in corrupt_data_keys:
                    audio_util.extract_audio_from_video_file(
                        original_video_file_path,
                        audio_file_path,
                        start=0,
                        length=-1)

                if C.DATA_KEY_AUDIO_SPECTROGRAMS in corrupt_data_keys:
                    spectrograms_dir = os.path.join(self._paths_for_data_key(C.DATA_KEY_AUDIO_SPECTROGRAMS)[0], fold)
                    spectogram_file_path = os.path.join(spectrograms_dir, f'{identifier}.npy')
                    video_clip = VideoFileClip(original_video_file_path)
                    video_length = video_clip.reader.nframes
                    wav_file = torchaudio.load(audio_file_path)[0]
                    if video_length == 1:
                        os.remove(original_video_file_path)
                        os.remove(audio_file_path)
                        continue

                    self._extract_audio_spectrogram(audio_file_path, spectogram_file_path)

                if C.DATA_KEY_TEXTS in corrupt_data_keys:
                    texts_dir = os.path.join(self._paths_for_data_key(C.DATA_KEY_TEXTS)[0], fold)
                    texts = all_subtitles[diag_id][utt_id]
                    text_file_path = os.path.join(texts_dir, f'{identifier}.txt')
                    with open(text_file_path, 'w') as texts_file:
                        texts_file.write(texts)

                if C.DATA_KEY_GLOVE_EMBEDDINGS in corrupt_data_keys:
                    texts = all_subtitles[diag_id][utt_id]
                    embeddings_dir = os.path.join(self._paths_for_data_key(C.DATA_KEY_GLOVE_EMBEDDINGS)[0], fold)
                    embedding_file_path = os.path.join(embeddings_dir, f'{identifier}.npy')
                    embeddings = glove.get_vecs_by_tokens(
                        tokens_split(texts), lower_case_backup=True).numpy()
                    np.save(embedding_file_path, embeddings)

        if C.DATA_KEY_ANNOTATIONS in corrupt_data_keys:
            with open(self._paths_for_data_key(C.DATA_KEY_ANNOTATIONS)[0], 'w+') as annotations_file:
                yaml.safe_dump(annotations, annotations_file)

    def _prepare_annotations_data(self, annotations):
        # Sentiment is from -1 to 1 but CUDA expects classes >= 0
        if self.target_annotation == C.BATCH_KEY_SENTIMENT:
            for subset in self.AVAILABLE_SUBSETS:
                for key, value in annotations[subset].items():
                    annotations[subset][key][C.BATCH_KEY_SENTIMENT] = value[C.BATCH_KEY_SENTIMENT] + 1
        return annotations

    def _compute_class_weights(self, subset, annotations):
        print(f'Computing weights for {subset} classes...')
        # Hard-coded weights form the ofifical website:
        # https://github.com/declare-lab/MELD#class-weights
        if self.target_annotation == C.BATCH_KEY_EMOTIONS:
            self.class_weights[subset] = [4.0, 15.0, 15.0, 3.0, 1.0, 6.0, 3.0]
        else:
            collected_sentiment = np.arange(self.num_classes)
            for key, value in annotations[subset].items():
                sentiment = value[C.BATCH_KEY_SENTIMENT]
                collected_sentiment[sentiment] += 1
            N = float(np.sum(collected_sentiment))
            class_weights = N / collected_sentiment
            class_weights /= sum(class_weights)
            self.class_weights[subset] =  class_weights
