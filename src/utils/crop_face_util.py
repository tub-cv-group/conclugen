import cv2
import os
from facenet_pytorch import MTCNN
import torch
import numpy as np
from shutil import copyfile
from moviepy.editor import VideoFileClip, concatenate_videoclips

from utils import file_util


def crop_video(input_video, output_video, x1, x2, y1, y2, video_type):
    video_caputre = cv2.VideoCapture(input_video)
    fps = video_caputre.get(cv2.CAP_PROP_FPS)
    success, frame_src = video_caputre.read()
    # Fix incorrect result video size
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, frame_src.shape[1])
    y2 = min(y2, frame_src.shape[0])
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    size = (x2-x1, y2-y1)
    fourcc = cv2.VideoWriter_fourcc(*video_type)  # FMP4 for avi
    video_write = cv2.VideoWriter(output_video, fourcc, fps, size)
    while success:
        frame_target = frame_src[y1:y2, x1:x2]  # (split_height, split_width)
        success, frame_src = video_caputre.read()
        video_write.write(frame_target)
    video_caputre.release()
    video_write.release()
    return True


def crop_video_with_frame(input_video, output_video, x1, x2, y1, y2, video_type, start_frame, end_frame):
    video_caputre = cv2.VideoCapture(input_video)
    fps = video_caputre.get(cv2.CAP_PROP_FPS)
    success, frame_src = video_caputre.read()
    # Fix incorrect result video size
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, frame_src.shape[1])
    y2 = min(y2, frame_src.shape[0])
    size = (x2-x1, y2-y1)
    fourcc = cv2.VideoWriter_fourcc(*video_type)  # FMP4 for avi
    video_write = cv2.VideoWriter(output_video, fourcc, fps, size)
    frames = []
    # Get all frames first
    while success:
        frames.append(frame_src)
        success, frame_src = video_caputre.read()
    end_frame = min(end_frame, len(frames))
    if end_frame - start_frame < 10:
        # We cannot verify this earlier since we need to read all frames first
        print('Video has less than 10 frames. Skipping.')
        return False
    for frame_src in frames[start_frame:end_frame]:
        frame_target = frame_src[y1:y2, x1:x2]
        video_write.write(frame_target)
    video_caputre.release()
    video_write.release()
    return True


def overlap(R1, R2):
    if (R1[0] > R2[2]) or (R1[2] < R2[0]) or (R1[3] < R2[1]) or (R1[1] > R2[3]):
        return False
    else:
        return True


mtcnn = None


def crop_video_to_face(
    video_file_path,
    video_target_path,
    video_type,
    batch_size=1
):
    """This function crops the video at `video_file_path` to the detected faces and stores the result at
    `video_target_path`. Because sometimes there is so little overlap between faces deteced in frames, this function
    will store individual videos. If you want the result to be merged, use `crop_to_face_merge_video` instead.

    Args:
        video_file_path (str): input video file path
        video_target_path (str): base path for the output video file
        video_type (str): the type of the video
        batch_size (int, optional): batch size to process the frames with. Defaults to 1.

    Returns:
        tuple: the coordinates, the number of output videos, the width and the height of the output videos
    """
    assert isinstance(batch_size, int) and batch_size > 0, f'Batch size must be a positive integer and not {batch_size}.'
    frame_idx = 0
    video_count = 0
    global mtcnn
    if mtcnn is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(keep_all=False, select_largest=False, device=device)
    # detector = MTCNN()
    video = cv2.VideoCapture(video_file_path)
    if not video.isOpened():
        raise Exception(f"Error reading video file path {video_file_path}.")
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_full = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if frames_full == 0:
        print(f'No frames found for video {video_file_path}. Skipping.')
        return [], 0, 0, 0
    x1min = frame_width
    y1min = frame_height
    x2max = 0
    y2max = 0
    prex1 = 0
    prex2 = 0
    prey1 = 0
    prey2 = 0
    start_frame = 0
    end_frame = frames_full
    coordinates = []
    facial_landmarks = []
    result_width = 0
    result_height = 0
    frame_idx = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    # For batched processing
    current_frame_batch = []
    current_frame_idx_batch = []
    exit_video_loop = False
    while (frame_idx <= frames_full - 1):
        ret, frame = video.read()
        # We will break if not ret at the end
        if ret:
            frame_idx = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            current_frame_batch.append(frame)
            current_frame_idx_batch.append(frame_idx)
            max_frame_height = frame.shape[0]
            max_frame_width = frame.shape[1]
        # The following checks if
        # 1. The batch is full (can only happen if ret is True since only then we append frames above)
        # 2. The last frame was not invalid and the batch is not empty, i.e. we process what we alread have accumulated
        # 3. We are at the end of the video and process the whole thing
        if len(current_frame_batch) == batch_size or\
            (not ret and len(current_frame_batch) > 0) or\
                frame_idx == frames_full - 1:
            if len(current_frame_batch) == 0:
                # Really weird case where none of the frames were valid, we just skip it for now
                print(f'No valid frames found for video {video_file_path}. Skipping.')
                break
            boxes, _, landmarks = mtcnn.detect(current_frame_batch, landmarks=True)
            for idx, (box, points) in enumerate(zip(boxes, landmarks)):
                _frame_idx = current_frame_idx_batch[idx]
                if type(box) == np.ndarray:
                    [x1, y1, x2, y2] = box[0]
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    if prex1 == 0 and prex2 == 0 and prey1 == 0 and prey2 == 0:
                        intersection = True
                        prex1 = x1
                        prex2 = x2
                        prey1 = y1
                        prey2 = y2
                    else:
                        intersection = overlap(
                            [x1, y1, x2, y2], [prex1, prey1, prex2, prey2])
                    # if intersection and frame_count<frames_full:
                    # if frame_count<frames_full-1:
                    if _frame_idx < frames_full - 1:
                        if intersection:
                            x1min = max(min(x1, x1min), 0)
                            y1min = max(min(y1, y1min), 0)
                            x2max = min(max(x2max, x2), max_frame_width)
                            y2max = min(max(y2max, y2), max_frame_height)
                            prex1 = x1
                            prex2 = x2
                            prey1 = y1
                            prey2 = y2
                        else:
                            if video_type == '.mp4':
                                fourcc = 'mp4v'
                            if video_type == '.avi':
                                fourcc = 'FMP4'
                            end_frame = _frame_idx
                            if end_frame - start_frame > 10:
                                video_count = video_count + 1
                                target_path = video_target_path.replace(
                                    video_type, '_' + str(video_count) + video_type)
                                success = crop_video_with_frame(
                                    video_file_path, target_path, x1min, x2max, y1min, y2max,
                                    fourcc, start_frame, end_frame)
                                if not success:
                                    video_count -= 1
                                    # This can be the case if e.g. less than 10 frames would be written. This is
                                    # verified in crop_video_with_frame where the acutal frames are loaded.
                                    if os.path.exists(target_path):
                                        os.remove(target_path)
                                start_frame = end_frame
                                coordinates.append(
                                    [x1min, x2max, y1min, y2max])
                                facial_landmarks.append(points)
                                width = x2max-x1min
                                height = y2max-y1min
                                result_width = max(width, result_width)
                                result_height = max(height, result_height)
                                x1min = frame_width
                                y1min = frame_height
                                x2max = 0
                                y2max = 0
                                prex1 = 0
                                prex2 = 0
                                prey1 = 0
                                prey2 = 0
                            else:
                                # Will continue the iteration over the detected boxes
                                print('Video has less than 10 frames. Skipping.')
                                continue
                if _frame_idx == frames_full - 1:
                    if video_type == '.mp4':
                        fourcc = 'mp4v'
                    if video_type == '.avi':
                        fourcc = 'FMP4'

                    end_frame = _frame_idx
                    if end_frame - start_frame > 10:
                        video_count = video_count + 1
                        target_path = video_target_path.replace(
                            video_type, '_' + str(video_count) + video_type)
                        if end_frame == frames_full - 1:
                            # In this case we have a cotinuous face detection until the end of the video
                            crop_video(
                                video_file_path, target_path, x1min, x2max, y1min, y2max, fourcc)
                        else:
                            success = crop_video_with_frame(
                                video_file_path, target_path, x1min, x2max, y1min,
                                y2max, fourcc, start_frame, end_frame)
                            if not success:
                                video_count -= 1
                                # This can be the case if e.g. less than 10 frames would be written. This is
                                # verified in crop_video_with_frame where the acutal frames are loaded.
                                if os.path.exists(target_path):
                                    os.remove(target_path)
                        coordinates.append([x1min, x2max, y1min, y2max])
                        facial_landmarks.append(points)
                        width = x2max-x1min
                        height = y2max-y1min
                        result_width = max(width, result_width)
                        result_height = max(height, result_height)
                    exit_video_loop = True
                    break
                if video_count > 30:
                    exit_video_loop = True
                    # At some point we just stop if there are more than 30 individual videos
                    break
            if exit_video_loop:
                break
        if len(current_frame_batch) == batch_size:
            current_frame_batch = []
            current_frame_idx_batch = []
        # If the last read frame was invalid, break
        if not ret:
            break
    video.release()
    if video_count == 0:
        print(f'No faces detected for video {video_file_path}.')
    return coordinates, facial_landmarks, video_count, result_width, result_height


def crop_to_face_merge_video(input_video, split_video, merged_video,
                             resize_scale=None, batch_size=1, with_landmarks=False):
    """This function crops the video at `input_video` to the detected faces and stores the result at
    `merged_video`. In case that there is too little overlap between detected face coordinates, the individual
    videos will first be stored using the base path `split_video` and the final merged result will be written
    to `merged_video`.

    If you provide a `resize_scale`, the final video will be resized to the given scale. You can also pass a
    `batch_size` to the function to speed up the detection process. Bear in mind that the videos are treated in
    their original resolution which calls for a lower batch size.

    Args:
        input_video (str): The input video.
        split_video (str): The base path for the split videos.
        merged_video (str): The path for the merged video.
        resize_scale (int, optional): Size to resize the video, will be transformed into a tuple. Defaults to None.
        batch_size (int, optional): The batch size to process the frames with. Defaults to 1.

    Returns:
        tuple: The number of split videos (even though there will be a merged result), the coordinates of the faces
    """
    assert isinstance(batch_size, int) and batch_size > 0, f'Batch size must be a positive integer and not {batch_size}.'
    coordinates, landmarks, count, x, y = crop_video_to_face(
        input_video, split_video, '.mp4', batch_size=batch_size)
    final_clip = None
    if count == 0:
        pass
    elif count == 1:
        filename = file_util.remove_extension_from_path(split_video) + '_1' + file_util.get_extension(split_video)
        clip = VideoFileClip(filename)
        if resize_scale is not None:
            clip = clip.resize((resize_scale, resize_scale))
        final_clip = clip
    else:
        videolist = []
        #final_clip = concatenate_videoclips([clip1, clip2], method='compose')
        for i in range(1, count+1):
            filename = file_util.remove_extension_from_path(split_video) + f'_{i}' + file_util.get_extension(split_video)
            clipnew = VideoFileClip(filename).resize((x, y))
            videolist.append(clipnew)
        concatenated_clip = concatenate_videoclips(videolist, method='compose')
        if resize_scale is not None:
            concatenated_clip = concatenated_clip.resize((resize_scale, resize_scale))
        final_clip = concatenated_clip
    if final_clip:
        final_clip.write_videofile(merged_video)
    if with_landmarks:
        return count, coordinates, landmarks
    return count, coordinates
