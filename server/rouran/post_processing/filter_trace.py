# -*- coding: utf-8 -*-
# @Time     : 2019/07/31 19:16
# @Author   : Bill H
# @Email    : lcurious@163.com
# @File     : filter_trace.py
# @License  : Apache-2.0
# Copyright (C) Bill H All rights reserved
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
import statsmodels.api as sm
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

from rouran.utils import utils
from rouran.utils.roi_tools import roi_coords2frame_coords


def get_labeled_index(df):
    # author: Xianming Xu, Qihan Huang, Zhenshuo Zhang
    labeled_index = [int(idx.split('/frame')[-1].split('.')[0]) for idx in df.index.to_list()]
    return labeled_index


def create_track(dt, R_std, Q_std, x):
    """
    The tracker is only for one point.
    The state variable we modeled is [x, vx, y, vy]^T
    The location (x, y) can be predicted by neural network
    Note that the x is the initial position
    :param dt:
    :param R_std:
    :param Q_std:
    :param x:
    :return:
    """
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    tracker.F = np.array([[1, dt, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, dt],
                          [0, 0, 0, 1]])
    tracker.u = 0.
    tracker.H = np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0]])
    tracker.R = np.eye(2) * R_std ** 2
    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std ** 2)
    tracker.Q = block_diag(q, q)
    tracker.x = np.array([[x[0], 0, x[1], 0]]).T
    tracker.p = np.eye(4) * 500
    return tracker


def kalman_post_process(labeled_df, estimated_df, estimated_roi_df):
    labeled_pts = labeled_df.values
    estimated_pts = estimated_df.values
    roi = estimated_roi_df.values
    labeled_idx = get_labeled_index(labeled_df)

    # replace the predicted frame with hand-labeled frame
    for i in range(len(labeled_idx)):
        estimated_pts[labeled_idx[i]] = labeled_pts[i]

    pts = estimated_pts.copy()
    for i in range(0, pts.shape[1], 2):
        pts[:, i] = pts[:, i] + roi[:, 0]
        pts[:, i + 1] = pts[:, i + 1] + roi[:, 1]

    # final output
    filter_pts = pts.copy()

    num_joints = pts.shape[1] // 2
    # parameter of KalmanFilter
    dt = 0.05
    R_std = 2
    Q_std = 10

    R = np.eye(2) * R_std ** 2
    Rs = [R for i in range(pts.shape[0])]
    for i in range(len(labeled_idx)):
        Rs[labeled_idx[i]] = 0

    for i in range(num_joints):
        zs = pts[:, 2 * i:2 * i + 2]
        x = pts[0, 2 * i:2 * i + 2]
        tracker = create_track(dt, R_std, Q_std, x)
        mu, _, _, _ = tracker.batch_filter(zs, Rs=Rs)

        filter_pts[:, 2 * i] = mu[:, 0, 0]
        filter_pts[:, 2 * i + 1] = mu[:, 2, 0]
    global_filtered_df = pd.DataFrame(data=filter_pts, columns=estimated_df.columns)

    # transform back: relative to the roi
    for i in range(0, pts.shape[1], 2):
        filter_pts[:, i] = filter_pts[:, i] - roi[:, 0]
        filter_pts[:, i + 1] = filter_pts[:, i + 1] - roi[:, 1]

    # write to file
    kalman_filtered_df = pd.DataFrame(data=filter_pts, columns=estimated_df.columns)
    return kalman_filtered_df, global_filtered_df


def filter_predictions(config_file, video, video_type='avi', overwrite_ask=True, shuffle=1, training_set_index=0,
                       filter_type='kalman',
                       window_length=5, p_bound=.001, AR_degree=2, MA_degree=2, alpha=.01, save_as_csv=True,
                       dest_folder=None):
    """

    Fits frame-by-frame pose predictions with ARIMA model (filtertype='arima') or median filter (default).

    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.

    video : string
        Full path of the video to extract the frame from. Make sure that this video is already analyzed.

    shuffle : int, optional
        The shufle index of training dataset. The extracted frames will be stored in the labeled-dataset for
        the corresponding shuffle of training dataset. Default is set to 1

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    filtertype: string
        Select which filter, 'arima' or 'median' filter.

    windowlength: int
        For filtertype='median' filters the input array using a local window-size given by windowlength. The array will automatically be zero-padded.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html The windowlenght should be an odd number.

    p_bound: float between 0 and 1, optional
        For filtertype 'arima' this parameter defines the likelihood below,
        below which a body part will be consided as missing data for filtering purposes.

    ARdegree: int, optional
        For filtertype 'arima' Autoregressive degree of Sarimax model degree.
        see https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

    MAdegree: int
        For filtertype 'arima' Moving Avarage degree of Sarimax model degree.
        See https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

    alpha: float
        Significance level for detecting outliers based on confidence interval of fitted SARIMAX model.

    save_as_csv: bool, optional
        Saves the predictions in a .csv file. The default is ``False``; if provided it must be either ``True`` or ``False``

    destfolder: string, optional
        Specifies the destination folder for analysis data (default is the path of the video). Note that for subsequent analysis this
        folder also needs to be passed.

    Returns filtered pandas array with the same structure as normal output of network.
    """
    cfg = utils.read_config(config_file)
    scorer = utils.get_model_name(cfg)
    videos = utils.get_list_of_videos(video, video_type)
    # get original video name
    pattern = re.compile(r'(.*)-roi-\d\.avi')
    roi_pattern = re.compile(r'.*-(roi-\d)\.avi')
    if len(videos) > 0:
        for video in videos:

            if dest_folder is None:
                dest_folder = str(Path(video).parents[0])

            print("Filtering with %s model %s" % (filter_type, video))
            video_folder = str(Path(video).parents[0])
            data_name = str(Path(video).stem) + '-' + scorer
            roi_offset_datafile = pattern.findall(video)[0] + '.csv'
            roi_name = roi_pattern.findall(video)[0]
            filtered_name = data_name + '-filtered.h5'
            # for make label video convenient [regex match convenient]
            filtered_global_name = data_name + '-filtered-Global.h5'
            if Path(os.path.join(video_folder, filtered_name)).exists() and overwrite_ask:
                usr_feedback = input("Video already filtered, do you want to overwrite it?(yes/[no])")
                if usr_feedback != 'yes' and usr_feedback != 'y':
                    continue
            try:
                coords_datafile = os.path.join(video_folder, data_name + '.h5')
                roi_offset_dataframe = pd.read_csv(roi_offset_datafile, header=[0, 1, 2], index_col=[0])
                roi_scorer = roi_offset_dataframe.columns.levels[0][0]
                coords_x, coords_y, joint_names = roi_coords2frame_coords(roi_offset_datafile,
                                                                          roi_name,
                                                                          coords_datafile)
                if filter_type == 'kalman':
                    labeled_file = os.path.join(cfg['project_path'], 'labeled-data')
                    labeled_file = os.path.join(labeled_file, Path(pattern.findall(video)[0]).stem)
                    labeled_file = os.path.join(labeled_file, roi_pattern.findall(video)[0])
                    labeled_file = os.path.join(labeled_file, 'CollectedData_' + cfg['scorer'] + '.csv')
                    # labeled_df = pd.read_hdf(str(labeled_file))
                    labeled_df = pd.read_csv(str(labeled_file), header=[0, 1, 2], index_col=[0])
                    estimated_df = pd.read_hdf(coords_datafile)
                    data, data_global = kalman_post_process(labeled_df, estimated_df, roi_offset_dataframe)
                else:
                    for bpindex, bp in enumerate(joint_names):
                        pdindex = pd.MultiIndex.from_product([[scorer], [bp], ['x', 'y']],
                                                             names=['scorer', 'bodyparts', 'coords'])
                        x, y = coords_x[bpindex], coords_y[bpindex]

                        if filter_type == 'arima':
                            mod_x = sm.tsa.statespace.SARIMAX(x.flatten(),
                                                              order=(AR_degree, 0, MA_degree),
                                                              seasonal_order=(0, 0, 0, 0),
                                                              simple_differencing=True)
                            mod_y = sm.tsa.statespace.SARIMAX(y.flatten(),
                                                              order=(AR_degree, 0, MA_degree),
                                                              seasonal_order=(0, 0, 0, 0),
                                                              simple_differencing=True)
                            res_x = mod_x.fit(disp=False)
                            res_y = mod_y.fit(disp=False)
                            predict_x = res_x.get_prediction(end=mod_x.nobs - 1)
                            predict_y = res_y.get_prediction(end=mod_y.nobs - 1)
                            mean_x, CIx = predict_x.predicted_mean, predict_x.conf_int(alpha=alpha)
                            mean_y, CIy = predict_y.predicted_mean, predict_y.conf_int(alpha=alpha)
                            mean_x[0] = x[0]
                            mean_y[0] = y[0]
                        else:
                            mean_x = signal.medfilt(x, kernel_size=window_length)
                            mean_y = signal.medfilt(y, kernel_size=window_length)

                        if bpindex == 0:
                            data_global = pd.DataFrame(np.hstack([np.expand_dims(mean_x, axis=1),
                                                                  np.expand_dims(mean_y, axis=1)]), columns=pdindex)
                            data = pd.DataFrame(np.hstack(
                                [np.expand_dims(mean_x - roi_offset_dataframe[roi_scorer][roi_name]['x-tl'].values,
                                                axis=1),
                                 np.expand_dims(mean_y - roi_offset_dataframe[roi_scorer][roi_name]['y-tl'].values,
                                                axis=1)]),
                                columns=pdindex)
                        else:
                            item_global = pd.DataFrame(np.hstack([np.expand_dims(mean_x, axis=1),
                                                                  np.expand_dims(mean_y, axis=1)]), columns=pdindex)
                            item = pd.DataFrame(np.hstack(
                                [np.expand_dims(mean_x - roi_offset_dataframe[roi_scorer][roi_name]['x-tl'].values,
                                                axis=1),
                                 np.expand_dims(mean_y - roi_offset_dataframe[roi_scorer][roi_name]['y-tl'].values,
                                                axis=1)]),
                                columns=pdindex)
                            data_global = pd.concat([data_global.T, item_global.T]).T
                            data = pd.concat([data.T, item.T]).T

                data_global.to_hdf(os.path.join(video_folder, filtered_global_name), 'filtered',
                                   format='table', mode='w')
                data.to_hdf(os.path.join(video_folder, filtered_name), 'filtered',
                            format='table', mode='w')
                if save_as_csv:
                    print("Saving filtered csv poses!")
                    data.to_csv(os.path.join(video_folder, filtered_name.split('.h5')[0] + '.csv'))
                    data_global.to_csv(os.path.join(video_folder, filtered_global_name.split('.h5')[0] + '.csv'))
            except FileNotFoundError as e:
                print("Video not analyzed -- Run analyze_videos first. {}".format(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('videos')
    cli_args = parser.parse_args()
