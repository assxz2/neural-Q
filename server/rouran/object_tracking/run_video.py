#  @Time     : 2019/08/21 16:11
#  @Author   : Bill H
#  @Email    : lcurious@163.com
#  @File     : run_video.py
#  @License  : Apache-2.0
#  Copyright (c) 2019. Bill H All rights reserved
#
#
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import pims
from easydict import EasyDict as edict
from tqdm import tqdm
import tifffile
import rouran
from rouran.utils import create_logger
from .models import models
from .tracker.siamfc import SiamFC
from .tracker.siamrpn import SiamRPN
from .utils.utils import load_pretrain, cxy_wh_2_rect

logger = create_logger("rouran.run_video")


def track_video(tracker, model, video_path, init_boxes=None):
    # assert os.path.isfile(video_path), "please provide a valid video file!"
    filename, file_extension = os.path.splitext(video_path)
    # if file_extension == '.tif':

    if file_extension in ['.avi', '.map4']:
        return track_video_norm(tracker, model, video_path, init_boxes)
    else:
        return track_video_tif(tracker, model, video_path, init_boxes)


def track_video_tif(tracker, model, video_path, init_boxes=None):
    frames = tifffile.imread(video_path)
    nframes = len(frames)
    frame = frames[0]
    if len(frame.shape) == 2:
        frame = np.stack([frame] * 3, axis=-1)

    # make tracking file
    multi_index = pd.MultiIndex.from_product([['SiamDW'],
                                              init_boxes.keys(),
                                              ['x-tl', 'y-tl', 'x-br', 'y-br']],
                                             names=['scorer', 'roi', 'coords'])
    coords = np.empty((nframes, 4 * len(init_boxes.keys())))
    coords[:] = np.nan
    if len(init_boxes.keys()):
        for ik, key in enumerate(init_boxes):
            lx, ly = init_boxes[key][0], init_boxes[key][1]
            w, h = init_boxes[key][2] - init_boxes[key][0], init_boxes[key][3] - init_boxes[key][1]
            target_pos = np.array([lx + w / 2, ly + h / 2])
            target_sz = np.array([w, h])
            state = tracker.init(frame, target_pos, target_sz, model)
            coords[0, ik * 4:ik*4+4] = [lx, ly, lx+w, ly+h]
            for n in tqdm(range(nframes - 1)):
                frame = frames[n]
                if len(frame.shape) == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                if frame is None:
                    continue
                frame_disp = frame.copy()

                state = tracker.track(state, frame_disp)
                location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                x1, y1, x2, y2 = location[0], location[1], location[0] + location[2], location[1] + location[3]
                coords[n+1, ik*4:ik*4+4] = [x1, y1, x2, y2]
        track_dataframe = pd.DataFrame(coords, columns=multi_index)
        return track_dataframe
    else:
        logger.error("ROI not provided!")
        return None


def track_video_norm(tracker, model, video_path, init_boxes=None):
    """
    generate csv and hdf file with track results and init_box should be a dict
    TODO: parse more rectangle ROI

    :param tracker:
    :param model:
    :param video_path:
    :param init_boxes:
        must be a dict with {"roi-1":[topLeftX, topLeftY, bottomRightX, bottomRightY]}
    :return:
    """

    assert os.path.isfile(video_path), "please provide a valid video file"

    cap = cv2.VideoCapture(video_path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, frame = cap.read()

    if success is not True:
        logger.error("Read failed.")
        exit(-1)

    # make tracking file
    multi_index = pd.MultiIndex.from_product([['SiamDW'],
                                              init_boxes.keys(),
                                              ['x-tl', 'y-tl', 'x-br', 'y-br']],
                                             names=['scorer', 'roi', 'coords'])
    coords = np.empty((nframes, 4 * len(init_boxes.keys())))
    coords[:] = np.nan
    # init
    if len(init_boxes.keys()):
        for ik, key in enumerate(init_boxes):
            print(key, init_boxes[key])
            cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
            lx, ly = init_boxes[key][0], init_boxes[key][1]
            w, h = init_boxes[key][2] - init_boxes[key][0], init_boxes[key][3] - init_boxes[key][1]
            target_pos = np.array([lx + w / 2, ly + h / 2])
            target_sz = np.array([w, h])
            state = tracker.init(frame, target_pos, target_sz, model)  # init tracker
            coords[0, ik * 4:ik * 4 + 4] = [lx, ly, lx + w, ly + h]
            for n in tqdm(range(nframes - 1)):
                ret, frame = cap.read()

                if frame is None:
                    continue
                frame_disp = frame.copy()

                # write to dict
                state = tracker.track(state, frame_disp)  # track
                location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                x1, y1, x2, y2 = location[0], location[1], location[0] + location[2], location[1] + location[3]
                coords[n + 1, ik * 4:ik * 4 + 4] = [x1, y1, x2, y2]
        track_dataframe = pd.DataFrame(coords, columns=multi_index)
        cap.release()
        return track_dataframe
    else:
        logger.error("ROI not provided!")
        return None


def post_process(data_frame):
    """
    make post process at dataframe

    :param data_frame:
    :return:
    """
    data_frame.rolling(2, axis=1)
    scorer = data_frame.columns.levels[0][0]
    roi_names = data_frame.columns.get_level_values(1).unique()
    for roi in roi_names:
        width = max(data_frame[scorer][roi]['x-br'] - data_frame[scorer][roi]['x-tl'])
        height = max(data_frame[scorer][roi]['y-br'] - data_frame[scorer][roi]['y-tl'])
        width_pad = (width - data_frame[scorer][roi]['x-br'] + data_frame[scorer][roi]['x-tl']) / 2
        height_pad = (height - data_frame[scorer][roi]['y-br'] + data_frame[scorer][roi]['y-tl']) / 2
        data_frame[scorer][roi]['x-tl'] -= width_pad
        data_frame[scorer][roi]['x-br'] += width_pad
        data_frame[scorer][roi]['y-tl'] -= height_pad
        data_frame[scorer][roi]['y-br'] += height_pad

    return data_frame


def track_roi_to_file(config, video, arch='SiamRPNRes22', resume='snapshot/CIResNet22_RPN.pth'):
    # prepare tracker
    info = edict()
    info.arch = arch
    info.dataset = video
    info.epoch_test = True
    info.cls_type = 'thinner'

    if 'FC' in arch:
        net = models.__dict__[arch]()
        tracker = SiamFC(info)
    else:
        net = models.__dict__[arch](anchors_nums=5, cls_type='thinner')
        tracker = SiamRPN(info)

    logger.info('[*] ======= Track video with {} ======='.format(arch))

    parent_path = Path(os.path.dirname(rouran.__file__))
    resume = str(Path(parent_path) / 'object_tracking' / resume)

    net = load_pretrain(net, resume)
    net.eval()
    net = net.cuda()

    # check init box is list or not
    if isinstance(video, list):
        video_list = video
    else:
        video_list = [video]

    first_frames = {}
    for video_file in video_list:
        # video_file may be symbol link
        file_name = config['video_sets'][video_file]['rois'].pop('file')
        init_boxes = config['video_sets'][video_file]['rois']
        dataframe = track_video(tracker, net, video_file, init_boxes)
        # dataframe should be post processed
        dataframe = post_process(dataframe)
        dataframe.to_csv(file_name)
    torch.cuda.empty_cache()


def tracking_once(video_path, init_boxes, predicted_csv_path):
    """ Make a tracking on specific video path and inti_box then write back tracking result to csv

    :param video_path: Original Video to track
    :type video_path: str
    :param init_boxes: {'Name': [topLeftX, topLeftY, bottomRightX, bottomRightY]}
    :type init_boxes: dict
    :param predicted_csv_path: predicted result save path
    :type predicted_csv_path: str
    :return:
    """
    # prepare tracker
    arch = 'SiamRPNRes22'
    resume = 'snapshot/CIResNet22_RPN.pth'

    info = edict()
    info.arch = arch
    info.dataset = video_path
    info.epoch_test = True
    info.cls_type = 'thinner'

    if 'FC' in arch:
        net = models.__dict__[arch]()
        tracker = SiamFC(info)
    else:
        net = models.__dict__[arch](anchors_nums=5, cls_type='thinner')
        tracker = SiamRPN(info)

    logger.info('[*] ======= Track video with {} ======='.format(arch))

    parent_path = Path(os.path.dirname(rouran.__file__))
    resume = str(Path(parent_path) / 'object_tracking' / resume)

    net = load_pretrain(net, resume)
    net.eval()
    net = net.cuda()

    # video_file may be symbol link
    dataframe = track_video(tracker, net, video_path, init_boxes)
    # dataframe should be post processed
    dataframe = post_process(dataframe)
    dataframe.to_csv(predicted_csv_path)
    torch.cuda.empty_cache()
    return True
