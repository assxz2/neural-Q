# -*- coding: utf-8 -*-
# @Time     : 2019/07/17 19:58
# @Author   : Bill H
# @Email    : lcurious@163.com
# @File     : roi_tools.py
# @License  : Apache-2.0
# Copyright (C) Bill H All rights reserved
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def roi_coords2frame_coords(roi_table_path, roi_index, roi_coords_path):
    """
    transform the coordinates in roi space into global frame space
    :param roi_table_path:
    :param roi_index:str
    :param roi_coords_path:
    :return:
    """
    data_frame = pd.read_hdf(roi_coords_path)
    roi_frame = pd.read_csv(roi_table_path, header=[0, 1, 2], index_col=[0])
    nframes = len(data_frame.index)

    assert len(roi_frame.columns.levels[0]) == 1, 'Multiple scorer parse Not Implements! Wait future!'

    roi_scorer = roi_frame.columns.levels[0][0]

    assert len(data_frame.columns.levels[0]) == 1, 'Multiple scorer export Not Implements! Wait future!'

    df_x = None
    df_y = None
    joint_names = []

    for model_name in data_frame.columns.levels[0]:
        joint_names = list(data_frame[model_name].columns.levels[0])
        jn_x = np.empty((len(joint_names), nframes))
        jn_y = np.empty((len(joint_names), nframes))
        for jn_index, jn in enumerate(joint_names):
            jn_x[jn_index, :] = data_frame[model_name][jn]['x'].values + \
                                roi_frame[roi_scorer][roi_index]['x-tl'].values
            jn_y[jn_index, :] = data_frame[model_name][jn]['y'].values + \
                                roi_frame[roi_scorer][roi_index]['y-tl'].values
            data_frame[model_name, jn, 'x'] += roi_frame[roi_scorer, roi_index, 'x-tl'].values
            data_frame[model_name, jn, 'y'] += roi_frame[roi_scorer, roi_index, 'y-tl'].values
        df_x = jn_x
        df_y = jn_y

    global_coords_table_name = str(Path(roi_coords_path).parent / str(Path(roi_coords_path).stem + '-Global'))
    # human read file
    data_frame.to_csv(global_coords_table_name + '.csv')
    # machine read file
    data_frame.to_hdf(global_coords_table_name + '.h5', 'coords_in_video')
    return df_x, df_y, joint_names


def make_global_labeled_video(video_path, roi_table_path, roi_index, roi_coords_path):
    cap = cv2.VideoCapture(video_path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    signal_table_name = str(Path(roi_coords_path).parent / Path(roi_coords_path).stem + 'Signal')
    wcap = cv2.VideoWriter('{}_labeled.avi'.format(os.path.splitext(video_path)[0]),
                           apiPreference=cv2.CAP_ANY,
                           fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps=10,
                           frameSize=(frame_width, frame_height))
    marker_x, marker_y, body_parts = roi_coords2frame_coords(roi_table_path, roi_index, roi_coords_path)

    energy = np.empty((nframes, len(body_parts)))
    index_length = int(np.ceil(np.log10(nframes)))
    for i in tqdm(range(nframes)):
        ret, frame = cap.read()
        for bp_index, bp in enumerate(body_parts):
            back = np.zeros_like(frame)
            cv2.circle(back, (int(marker_x[bp_index, i]), int(marker_y[bp_index, i])), 12, (50, 200, 50), -1)
            # compute energy
            energy[i, bp_index] = np.sum(cv2.bitwise_and(frame, back))
            cv2.circle(frame, (int(marker_x[bp_index, i]), int(marker_y[bp_index, i])), 12, (50, 200, 50), 2)
        cv2.imwrite(signal_table_name + '/img'+str(i).zfill(index_length)+'.png'.format(i), frame)
        wcap.write(frame)
    energy_data_frame = pd.DataFrame(data=energy, columns=body_parts)
    # human read file
    energy_data_frame.to_csv(signal_table_name + '.csv')
    # machine read file
    energy_data_frame.to_csv(signal_table_name + '.h5', 'firing_signal')
    wcap.release()
    cap.release()


if __name__ == '__main__':
    video_path = '/media/Develop/GridLight/rouran/examples/Tracing-PER-Bill-2019-07-31/videos/20180717_4_.avi'
    roi_path = '/media/Develop/GridLight/rouran/examples/Tracing-PER-Bill-2019-08-11/videos/20180717_4_.csv'
    hdf_path = '/media/Develop/GridLight/rouran/examples/Tracing-PER-Bill-2019-08-11/videos/20180717_4_-roi-1-HRNetV2-W18.h5'
    roi_coords2frame_coords(roi_path, 'roi-1', hdf_path)
    # collect_signal(video_path, roi_path, 0, hdf_path)
