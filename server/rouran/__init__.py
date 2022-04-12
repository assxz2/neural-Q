# -*- coding: utf-8 -*-
# @Time     : 2019/07/15 14:29
# @Author   : Huang Zenan
# @Email    : lcurious@163.com
# @File     : __init__.py.py
# @License  : Apache-2.0
# Copyright (C) Huang Zenan All rights reserved
from rouran.create_workspace import create_new_workspace
from rouran.object_tracking.run_video import tracking_once
from rouran.utils import make_roi_video, crop_videos, get_roi_file, make_single_roi_clip
from rouran.generate_training_dataset.extract_frame import extract_frames
from rouran.generate_training_dataset import create_training_dataset, label_frames, ANNOTATOR
from rouran.landmark_estimation import train_network, make_estimation, evaluate_network, create_labeled_video, \
    make_landmark_inference
from rouran.post_processing import filter_predictions
from rouran.utils import collect_signal, create_logger

logger = create_logger('rouran')
