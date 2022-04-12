# -*- coding: utf-8 -*-
#  @Time     : 2019/08/15 17:18
#  @Author   : Bill H
#  @Email    : lcurious@163.com
#  @File     : signal_collect.py
#  @License  : Apache-2.0
#  Copyright (c) 2019. Bill H All rights reserved
#
import re
from pathlib import Path
import tifffile
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import natsort
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
from rouran.utils import utils
# import matplotlib.pyplot as plt


def collect_signal_old(config_file, video_path, video_type='.avi', dest_folder=None):
    cfg = utils.read_config(config_file)
    scorer = utils.get_model_name(cfg)
    videos = utils.get_list_of_videos(video_path, video_type)
    signal_files = []
    if len(videos) > 0:
        for video_path in tqdm(videos):
            print(video_path)
            roi_coords_path = str(Path(video_path).parent / Path(video_path).stem) + '-' + scorer + '-filtered.h5'
            # Not match
            if not Path(roi_coords_path).exists():
                print("Video {} has not filtered!".format(video_path))
                break

            cap = cv2.VideoCapture(video_path)
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            signal_table_name = str(Path(roi_coords_path).parent / Path(roi_coords_path).stem) + 'Signal'

            coords_dataframe = pd.read_hdf(roi_coords_path, 'filtered')
            joint_names = cfg['bodyparts']
            energy = np.empty((nframes, len(joint_names)))
            for i in tqdm(range(nframes)):
                ret, frame = cap.read()
                for jn_index, jn in enumerate(joint_names):
                    jn_x = np.round(coords_dataframe.iloc[i][scorer][jn]['x'])
                    jn_y = np.round(coords_dataframe.iloc[i][scorer][jn]['y'])
                    back = np.zeros_like(frame)
                    cv2.circle(back, (int(jn_x), int(jn_y)), int(cfg['dotsize']), (255, 255, 255), -1)
                    bg_part = frame[:2, :2] + frame[-2:, :2] + frame[-2:, -2:] + frame[:2, -2:]
                    bg_val = np.sum(bg_part) / 16
                    front_pixels_num = np.sum(back > 0) / 3
                    # compute energy
                    energy[i, jn_index] = np.sum(cv2.bitwise_and(frame, back)) - bg_val*front_pixels_num
            energy_data_frame = pd.DataFrame(data=energy, columns=joint_names)
            # human read file
            energy_data_frame.to_csv(signal_table_name + '.csv')
            energy_data_frame.to_csv(video_path[:-4] + '_signals.csv')
            signal_files.append(signal_table_name + '.csv')
            # energy_data_frame.to_excel(signal_table_name + '.xlsx', sheet_name='Neuron Firing')
            # machine read file
            energy_data_frame.to_hdf(signal_table_name + '.h5', 'firing_signal')
            print("=> Signal file created: {}".format(signal_table_name))
            cap.release()
    return signal_files

def collect_signal(file_path, tif_image, signal_path, line_path):
    img = tifffile.imread(tif_image)
    skeleton_df = pd.read_csv(file_path, header=[0, 1, 2],
                              index_col=0)
    mindex = pd.MultiIndex.from_tuples(skeleton_df.columns,
                                       names=['model_name', 'joint_name', 'coords'])
    frame_num = len(skeleton_df)
    signal_matrix = np.zeros((frame_num, len(mindex.levels[1])))
    for irow, row in skeleton_df.iterrows():
        gray = img[irow]
        for icol, index in enumerate(natsort.natsorted(mindex.levels[1])):
            xy_pt = row[mindex.levels[0][0], index]
            x = xy_pt['x']
            y = xy_pt['y']
            sx = np.clip(x, a_min=0, a_max=gray.shape[1])
            sy = np.clip(y, a_min=0, a_max=gray.shape[0])
            back = np.zeros_like(gray, dtype=np.uint8)
            cv2.circle(back, (int(sx), int(sy)), 12, 1, -1)
            pixel_num = np.sum(back)
            bg_part = gray[:2, :2] + gray[-2:, :2] + gray[-2:, -2:] + gray[:2, -2:]
            bg_val = np.mean(bg_part) / 4
            #np.sum(cv2.bitwise_and(gray, gray, mask=back))
            signal_matrix[irow, icol] = np.sum(cv2.bitwise_and(gray, gray, mask=back)) / pixel_num - bg_val

    signal_df = pd.DataFrame(columns=natsort.natsorted(mindex.levels[1]), data=signal_matrix)
    signal_df.to_csv(signal_path)
    with PdfPages(line_path) as pdf:
        signal_df.plot(subplots=True, figsize=(16, 16))
        pdf.savefig()
        plt.close()
    return 'Analysis Success'