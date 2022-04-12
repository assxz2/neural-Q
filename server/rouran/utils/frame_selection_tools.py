# -*- coding: utf-8 -*-
# @Time     : 2019/07/16 18:55
# @Author   : Bill H
# @Email    : lcurious@163.com
# @File     : frame_selection_tools.py
# @License  : Apache-2.0
# Copyright (C) Bill H All rights reserved
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from skimage.util import img_as_ubyte
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from rouran.utils import utils
from rouran.object_tracking import track_roi_to_file


def uniform_frames(clip, numframes2pick, start, stop, Index=None):
    ''' Temporally uniformly sampling frames in interval (start,stop).
    Visual information of video is irrelevant for this method. This code is fast and sufficient (to extract distinct frames),
    when behavioral videos naturally covers many states.

    The variable indexes allows to pass on a subindex for the frames.
    '''
    print("Uniformly extracting of frames from", round(start * clip.duration, 2), " seconds to",
          round(stop * clip.duration, 2), " seconds.")
    if Index is None:
        if start == 0:
            frames2pick = np.random.choice(math.ceil(clip.duration * clip.fps * stop), size=numframes2pick,
                                           replace=False)
        else:
            frames2pick = np.random.choice(
                range(math.floor(start * clip.duration * clip.fps), math.ceil(clip.duration * clip.fps * stop)),
                size=numframes2pick, replace=False)
        return frames2pick
    else:
        startindex = int(np.floor(clip.fps * clip.duration * start))
        stopindex = int(np.ceil(clip.fps * clip.duration * stop))
        Index = np.array(Index, dtype=np.int)
        Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!
        if len(Index) >= numframes2pick:
            return list(np.random.permutation(Index)[:numframes2pick])
        else:
            return list(Index)


# uses openCV
def uniform_frames_cv2(cap, numframes2pick, start, stop, Index=None):
    """
    Temporally uniformly sampling frames in interval (start,stop).
    Visual information of video is irrelevant for this method. This code is fast and sufficient (to extract distinct frames),
    when behavioral videos naturally covers many states.

    The variable indexes allows to pass on a subindex for the frames.
    """
    nframes = int(cap.get(7))
    print("Uniformly extracting of frames from", round(start * nframes * 1. / cap.get(5), 2), " seconds to",
          round(stop * nframes * 1. / cap.get(5), 2), " seconds.")

    if Index is None:
        if start == 0:
            frames2pick = np.random.choice(math.ceil(nframes * stop), size=numframes2pick, replace=False)
        else:
            frames2pick = np.random.choice(range(math.floor(nframes * start), math.ceil(nframes * stop)),
                                           size=numframes2pick, replace=False)
        return frames2pick
    else:
        startindex = int(np.floor(nframes * start))
        stopindex = int(np.ceil(nframes * stop))
        Index = np.array(Index, dtype=np.int)
        Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!
        if len(Index) >= numframes2pick:
            return list(np.random.permutation(Index)[:numframes2pick])
        else:
            return list(Index)


def Kmeans_based_frame_selection(clip, numframes2pick, start, stop, Index=None, step=1, resizewidth=30, batchsize=100,
                                 max_iter=50, color=False):
    ''' This code downsamples the video to a width of resizewidth.

    The video is extracted as a numpy array, which is then clustered with kmeans, whereby each frames is treated as a vector.
    Frames from different clusters are then selected for labeling. This procedure makes sure that the frames "look different",
    i.e. different postures etc. On large videos this code is slow.

    Consider not extracting the frames from the whole video but rather set start and stop to a period around interesting behavior.

    Note: this method can return fewer images than numframes2pick.'''

    print("Kmeans-quantization based extracting of frames from", round(start * clip.duration, 2), " seconds to",
          round(stop * clip.duration, 2), " seconds.")
    startindex = int(np.floor(clip.fps * clip.duration * start))
    stopindex = int(np.ceil(clip.fps * clip.duration * stop))

    if Index is None:
        Index = np.arange(startindex, stopindex, step)
    else:
        Index = np.array(Index)
        Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!

    nframes = len(Index)
    if batchsize > nframes:
        batchsize = int(nframes / 2)

    if len(Index) >= numframes2pick - 1:
        clipresized = clip.resize(width=resizewidth)
        ny, nx = clipresized.size
        frame0 = img_as_ubyte(clip.get_frame(0))
        if np.ndim(frame0) == 3:
            ncolors = np.shape(frame0)[2]
        else:
            ncolors = 1
        print("Extracting and downsampling...", nframes, " frames from the video.")

        if color and ncolors > 1:
            DATA = np.zeros((nframes, nx * 3, ny))
            for counter, index in enumerate(tqdm(Index)):
                image = img_as_ubyte(clipresized.get_frame(index * 1. / clipresized.fps))
                DATA[counter, :, :] = np.vstack([image[:, :, 0], image[:, :, 1], image[:, :, 2]])
        else:
            DATA = np.zeros((nframes, nx, ny))
            for counter, index in enumerate(tqdm(Index)):
                if ncolors == 1:
                    DATA[counter, :, :] = img_as_ubyte(clipresized.get_frame(index * 1. / clipresized.fps))
                else:
                    # attention: averages over color channels to keep size small / perhaps you want to
                    # use color information?
                    DATA[counter, :, :] = img_as_ubyte(
                        np.array(np.mean(clipresized.get_frame(index * 1. / clipresized.fps), 2), dtype=np.uint8))

        print("Kmeans clustering ... (this might take a while)")
        data = DATA - DATA.mean(axis=0)
        data = data.reshape(nframes, -1)  # stacking

        kmeans = MiniBatchKMeans(n_clusters=numframes2pick, tol=1e-3, batch_size=batchsize, max_iter=max_iter)
        kmeans.fit(data)
        frames2pick = []
        for cluster_id in range(numframes2pick):  # pick one frame per cluster
            cluster_ids = np.where(cluster_id == kmeans.labels_)[0]

            num_images_of_cluster = len(cluster_ids)
            if num_images_of_cluster > 0:
                frames2pick.append(Index[cluster_ids[np.random.randint(num_images_of_cluster)]])

        clipresized.close()
        del clipresized
        return list(np.array(frames2pick))
    else:
        return list(Index)


def Kmeans_based_frame_selection_cv2(cap, numframes2pick, start, stop, indexes=None, step=1, resizewidth=30,
                                     batchsize=100, max_iter=50, color=False):
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    ny = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ratio = resizewidth * 1. / nx
    if ratio > 1:
        raise Exception("Chose of resize width actually upsamples!")

    print("Kmeans-quantization based extracting of frames from", round(start * nframes * 1. / cap.get(5), 2),
          " seconds to", round(stop * nframes * 1. / cap.get(5), 2), " seconds.")
    startindex = int(np.floor(nframes * start))
    stopindex = int(np.ceil(nframes * stop))

    if indexes is None:
        indexes = np.arange(startindex, stopindex, step)
    else:
        indexes = np.array(indexes)
        indexes = indexes[(indexes > startindex) * (indexes < stopindex)]  # crop to range!

    nframes = len(indexes)
    if batchsize > nframes:
        batchsize = int(nframes / 2)

    allocated = False
    if len(indexes) >= numframes2pick - 1:
        if np.mean(np.diff(indexes)) > 1:
            # then non-consecutive indices are present, thus cap.set is required (which slows everything down!)
            print("Extracting and downsampling...", nframes, " frames from the video.")
            if color:
                for counter, index in enumerate(tqdm(indexes)):
                    cap.set(1, index)  # extract a particular frame
                    ret, frame = cap.read()
                    if ret:
                        # image=img_as_ubyte(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),None,fx=ratio,fy=ratio))
                        # color trafo not necessary; lack thereof improves speed.
                        image = img_as_ubyte(cv2.resize(frame, None, fx=ratio, fy=ratio,
                                                        interpolation=cv2.INTER_NEAREST))
                        if not allocated:  # 'DATA' not in locals(): #allocate memory in first pass
                            DATA = np.empty((nframes, np.shape(image)[0], np.shape(image)[1] * 3))
                            allocated = True
                        DATA[counter, :, :] = np.hstack([image[:, :, 0], image[:, :, 1], image[:, :, 2]])
            else:
                for counter, index in enumerate(tqdm(indexes)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)  # extract a particular frame
                    ret, frame = cap.read()
                    if ret:
                        # image=img_as_ubyte(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),None,fx=ratio,fy=ratio))
                        # color trafo not necessary; lack thereof improves speed.
                        image = img_as_ubyte(cv2.resize(frame, None, fx=ratio, fy=ratio,
                                                        interpolation=cv2.INTER_NEAREST))
                        if not allocated:  # 'DATA' not in locals(): #allocate memory in first pass
                            DATA = np.empty((nframes, np.shape(image)[0], np.shape(image)[1]))
                            allocated = True
                        DATA[counter, :, :] = np.mean(image, 2)
        else:
            print("Extracting and downsampling...", nframes, " frames from the video.")
            if color:
                for counter, index in enumerate(tqdm(indexes)):
                    ret, frame = cap.read()
                    if ret:

                        # image=img_as_ubyte(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),None,fx=ratio,fy=ratio))
                        # color trafo not necessary; lack thereof improves speed.
                        image = img_as_ubyte(cv2.resize(frame, None, fx=ratio, fy=ratio,
                                                        interpolation=cv2.INTER_NEAREST))
                        if not allocated:  # 'DATA' not in locals(): #allocate memory in first pass
                            DATA = np.empty((nframes, np.shape(image)[0], np.shape(image)[1] * 3))
                            allocated = True
                        DATA[counter, :, :] = np.hstack([image[:, :, 0], image[:, :, 1], image[:, :, 2]])
            else:
                for counter, index in enumerate(tqdm(indexes)):
                    ret, frame = cap.read()
                    if ret:
                        # image=img_as_ubyte(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),None,fx=ratio,fy=ratio))
                        # color trafo not necessary; lack thereof improves speed.
                        image = img_as_ubyte(cv2.resize(frame, None, fx=ratio, fy=ratio,
                                                        interpolation=cv2.INTER_NEAREST))
                        if not allocated:  # 'DATA' not in locals(): #allocate memory in first pass
                            DATA = np.empty((nframes, np.shape(image)[0], np.shape(image)[1]))
                            allocated = True
                        DATA[counter, :, :] = np.mean(image, 2)

        print("Kmeans clustering ... (this might take a while)")
        data = DATA - DATA.mean(axis=0)
        data = data.reshape(nframes, -1)  # stacking

        kmeans = MiniBatchKMeans(n_clusters=numframes2pick, tol=1e-3, batch_size=batchsize, max_iter=max_iter)
        kmeans.fit(data)
        frames2pick = []
        for clusterid in range(numframes2pick):  # pick one frame per cluster
            clusterids = np.where(clusterid == kmeans.labels_)[0]

            num_images_of_cluster = len(clusterids)
            if num_images_of_cluster > 0:
                frames2pick.append(indexes[clusterids[np.random.randint(num_images_of_cluster)]])
        # cap.release() >> still used in frame_extraction!
        return list(np.array(frames2pick))
    else:
        return list(indexes)


def make_roi_video(video_path, roi_table_path, exist_ask=True):
    """
    crop every frame according to roi description in roi_info table,
    use biggest roi width and height along the sequence
    :param exist_ask:
    :param video_path: string, path to real video path
    :param roi_table_path: string, path of roi description
    :return:
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # parse the roi information
    roi_table = pd.read_csv(roi_table_path, header=[0, 1, 2], index_col=[0])
    roi_bboxs = []
    roi_list = []
    frame_widths = []
    frame_heights = []
    # https://stackoverflow.com/questions/25929319/how-to-iterate-over-pandas-multiindex-dataframe-using-index
    for scorer in roi_table.columns.get_level_values(0).unique():
        roi = roi_table[scorer].columns.get_level_values(0).unique()[0]
        # print("Extracting user {} defined {} ROI ...".format(scorer, roi))
        roi_video_frames = 0
        if os.path.exists(roi_clip_path):
            roi_cap = cv2.VideoCapture(roi_clip_path)
            roi_video_frames = int(roi_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            roi_cap.release()

        if roi_video_frames == nframes and exist_ask:
            usr_feedback = input("Video ROI: {} already cropped! Do you want overwrite it?(yes/[no])".format(roi))
            if usr_feedback != 'yes' or usr_feedback != 'y':
                continue
        bbox = np.around(np.asarray(roi_table[scorer][roi])).astype(np.int)
        roi_bboxs.append(bbox)
        frame_width = np.max(bbox[:, 2] - bbox[:, 0])
        frame_height = np.max(bbox[:, 3] - bbox[:, 1])
        frame_widths.append(frame_width)
        frame_heights.append(frame_height)
        out_videos = tifffile.TiffWriter(roi_clip_path)

    if len(roi_bboxs):
        for i in tqdm(range(nframes)):
            frame = frames[i]
            height, width = frame.shape[0], frame.shape[1]

            if frame is None:
                print("Frame {} lost!".format(i))
                continue
            for vid, roi_bbox in enumerate(roi_bboxs):
                roi_bbox[roi_bbox < 0] = 0
                xtl, ytl, xbr, ybr = roi_bbox[i, :]
                roi_frame = np.zeros((frame_heights[vid], frame_widths[vid]), dtype=frame.dtype)
                h_pad = min(frame_heights[vid], height - ytl)
                w_pad = min(frame_widths[vid], width - xtl)
                try:
                    roi_frame[:h_pad, :w_pad] = frame[ytl:ytl + frame_heights[vid], xtl:xtl + frame_widths[vid]]
                    roi_list.append(roi_frame)
                except ValueError as ve:
                    pass
        out_videos.save(roi_list, dtype=frame.dtype)
        #         for wcap in out_videos:
        #             wcap.close()



def make_single_roi_clip(video_path, roi_table_path, roi_clip_path, exist_ask=True):
    filename, file_extension = os.path.splitext(roi_clip_path)
    if roi_clip_path[-4:] == '.tif':
        return make_single_roi_tif_clip(video_path, roi_table_path, roi_clip_path, exist_ask)
    else:
        return make_single_roi_video_clip(video_path, roi_table_path, roi_clip_path, exist_ask)


def make_single_roi_video_clip(video_path, roi_table_path, roi_clip_path, exist_ask=True):
    """Crop every frame according to roi description in roi_info table,
    use biggest roi width and height along the sequence

    :param exist_ask:
    :param video_path: string, path to real video path
    :param roi_table_path: string, path of roi description
    :return:
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # parse the roi information
    roi_table = pd.read_csv(roi_table_path, header=[0, 1, 2], index_col=[0])
    roi_bboxs = []
    out_videos = []
    frame_widths = []
    frame_heights = []
    # https://stackoverflow.com/questions/25929319/how-to-iterate-over-pandas-multiindex-dataframe-using-index
    for scorer in roi_table.columns.get_level_values(0).unique():
        roi = roi_table[scorer].columns.get_level_values(0).unique()[0]
        # print("Extracting user {} defined {} ROI ...".format(scorer, roi))
        roi_video_frames = 0
        if os.path.exists(roi_clip_path):
            roi_cap = cv2.VideoCapture(roi_clip_path)
            roi_video_frames = int(roi_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            roi_cap.release()

        if roi_video_frames == nframes and exist_ask:
            usr_feedback = input("Video ROI: {} already cropped! Do you want overwrite it?(yes/[no])".format(roi))
            if usr_feedback != 'yes' or usr_feedback != 'y':
                continue
        bbox = np.around(np.asarray(roi_table[scorer][roi])).astype(np.int)
        roi_bboxs.append(bbox)
        frame_width = np.max(bbox[:, 2] - bbox[:, 0])
        frame_height = np.max(bbox[:, 3] - bbox[:, 1])
        frame_widths.append(frame_width)
        frame_heights.append(frame_height)
        out_videos.append(cv2.VideoWriter(roi_clip_path,
                                          apiPreference=cv2.CAP_ANY,
                                          fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps=fps,
                                          frameSize=(frame_width, frame_height)))

    if len(roi_bboxs):
        for i in tqdm(range(nframes)):
            ret, frame = cap.read()
            height, width = frame.shape[0], frame.shape[1]
            if frame is None:
                print("Frame {} lost!".format(i))
                continue
            for vid, roi_bbox in enumerate(roi_bboxs):
                roi_bbox[roi_bbox < 0] = 0
                xtl, ytl, xbr, ybr = roi_bbox[i, :]
                roi_frame = np.zeros((frame_heights[vid], frame_widths[vid], 3), dtype=np.uint8)
                h_pad = min(frame_heights[vid], height - ytl)
                w_pad = min(frame_widths[vid], width - xtl)
                try:
                    roi_frame[:h_pad, :w_pad] = frame[ytl:ytl+frame_heights[vid], xtl:xtl+frame_widths[vid], :]
                except ValueError as ve:
                    pass
                out_videos[vid].write(roi_frame)
    for wcap in out_videos:
        wcap.release()
    return True


def make_single_roi_tif_clip(video_path, roi_table_path, roi_clip_path, exist_ask=True):
    frames = tifffile.imread(video_path)
    nframes = len(frames)
    roi_table = pd.read_csv(roi_table_path, header=[0, 1, 2], index_col=[0])
    roi_bboxs = []
    roi_list = []
    frame_widths = []
    frame_heights = []
    # https://stackoverflow.com/questions/25929319/how-to-iterate-over-pandas-multiindex-dataframe-using-index
    for scorer in roi_table.columns.get_level_values(0).unique():
        roi = roi_table[scorer].columns.get_level_values(0).unique()[0]
        # print("Extracting user {} defined {} ROI ...".format(scorer, roi))
        roi_video_frames = 0
        if os.path.exists(roi_clip_path):
            roi_cap = cv2.VideoCapture(roi_clip_path)
            roi_video_frames = int(roi_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            roi_cap.release()

        if roi_video_frames == nframes and exist_ask:
            usr_feedback = input("Video ROI: {} already cropped! Do you want overwrite it?(yes/[no])".format(roi))
            if usr_feedback != 'yes' or usr_feedback != 'y':
                continue
        bbox = np.around(np.asarray(roi_table[scorer][roi])).astype(np.int)
        roi_bboxs.append(bbox)
        frame_width = np.max(bbox[:, 2] - bbox[:, 0])
        frame_height = np.max(bbox[:, 3] - bbox[:, 1])
        frame_widths.append(frame_width)
        frame_heights.append(frame_height)
        out_videos = tifffile.TiffWriter(roi_clip_path)

    if len(roi_bboxs):
        for i in tqdm(range(nframes)):
            frame = frames[i]
            height, width = frame.shape[0], frame.shape[1]

            if frame is None:
                print("Frame {} lost!".format(i))
                continue
            for vid, roi_bbox in enumerate(roi_bboxs):
                roi_bbox[roi_bbox < 0] = 0
                xtl, ytl, xbr, ybr = roi_bbox[i, :]
                roi_frame = np.zeros((frame_heights[vid], frame_widths[vid]), dtype=frame.dtype)
                h_pad = min(frame_heights[vid], height - ytl)
                w_pad = min(frame_widths[vid], width - xtl)
                try:
                    roi_frame[:h_pad, :w_pad] = frame[ytl:ytl + frame_heights[vid], xtl:xtl + frame_widths[vid]]
                    roi_list.append(roi_frame)
                except ValueError as ve:
                    pass
        out_videos.save(roi_list, dtype=frame.dtype)
        #         for wcap in out_videos:
        #             wcap.close()
        return True


def get_roi_file(config_file, ask=True, video_file=None):
    config = utils.read_config(config_file)
    if video_file is None:
        video_sets = config['video_sets']
    else:
        video_sets = {video_file: config['video_sets'][video_file]}
    # make the roi label path

    to_track_videos = []

    for video_name in video_sets.keys():
        video_roi_file = video_sets[video_name]['rois']['file']
        if Path(video_roi_file).exists():
            if ask:
                usr_feedback = input("ROI already been estimated, do you want to overwrite? (yes/[no])")
                if usr_feedback != 'yes' and usr_feedback != 'y':
                    continue
        # Filter the uninitialized tasks
        if len(video_sets[video_name]['rois']['roi-1']):
            to_track_videos.append(video_name)

    if len(to_track_videos):
        track_roi_to_file(config, to_track_videos)


def crop_videos(config_file, video_file=None, exist_ask=True):
    """
    crop all videos in the workspace if the corresponding roi csv description file exits
    :param exist_ask:
    :param video_file:
    :param config_file:
    :return:
    """
    config = utils.read_config(config_file)
    if video_file is None:
        video_sets = config['video_sets']
    else:
        video_sets = {video_file: config['video_sets'][video_file]}
    # make the roi label path
    project_path = Path(config['project_path'])
    labeled_path = project_path / 'labeled-data'

    for key in video_sets.keys():
        try:
            video_roi_table_path = video_sets[key]['rois']['file']
        except:
            print(video_sets[key]['rois']['file'], " Not found! Please run rouran.get_roi_file before!")
            continue

        roi_table = pd.read_csv(video_roi_table_path, header=[0, 1, 2])
        video_labeled_path = labeled_path / Path(key).stem
        # make video path, each roi will get its own folder
        for scorer in roi_table.columns.get_level_values(0).unique()[1:]:
            for roi in roi_table[scorer].columns.get_level_values(0).unique():
                utils.attempt_to_make_folder(video_labeled_path / roi)
        print("Cropping video [{}] according to [{}]".format(key, video_roi_table_path))
        make_roi_video(key, video_roi_table_path, exist_ask)


if __name__ == '__main__':
    video_cap = cv2.VideoCapture(
        '/media/Develop/GridLight/rouran/examples/Tracing-Bill-2019-07-16/videos/20180829_4-1.avi')
    make_roi_video(video_cap,
                   '/media/Develop/GridLight/rouran/examples/Tracing-Bill-2019-07-16/videos/20180829_4-1.csv')
