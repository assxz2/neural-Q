# -*- coding: utf-8 -*-
# @Time     : 2019/11/04 14:51
# @Author   : Bill H
# @Email    : lcurious@163.com
# @File     : create_demo.py
# @License  : Apache-2.0
# Copyright (C) Bill H All rights reserved
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pandas as pd
from itertools import groupby
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import cv2

sns.set(style="darkgrid")


def update_curve(num, data_dict, curve_axes):
    for i, key in enumerate(data_dict):
        group_data = data_dict[key]
        for sk in group_data:
            single_data = group_data[sk]
            curve_axes[i].set_data(single_data[:, :num])
    return curve_axes


def generate_curve(file_name):
    data_frame = pd.read_hdf(file_name)
    neuron_names = data_frame.columns.to_list()

    colors = plt.cm.get_cmap('jet', len(neuron_names))
    colors_dict = {}
    for i, n in enumerate(neuron_names):
        colors_dict[n] = colors(i)

    neuron_names.sort()
    group_names = [list(key) for i, key in groupby(neuron_names, lambda f: f.split('-')[0])]

    fig, axes = plt.subplots(len(group_names), sharex=True, sharey=True, figsize=(8, 8))

    xdata = data_frame.index.values
    curve_dict = {}
    for i, gn in enumerate(group_names):
        for sn in gn:
            ydata = data_frame[sn]
            curve_dict[sn] = axes[i].plot(xdata, ydata, c=colors_dict[sn])[0]
        axes[i].set_yticks([])
        axes[i].set_xlim([0, xdata.shape[0]])
        axes[i].legend(gn, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    for t in tqdm(range(xdata.shape[0])):
        for i, gn in enumerate(group_names):
            for sn in gn:
                ydata = data_frame[sn]
                curve_dict[sn].set_data(xdata[:t], ydata[:t])
        plt.savefig('/media/Develop/GridLight/Neural-Quantification/examples/signals/img{:0>3}.png'.format(t))


def draw_roi(video_file, roi_file):
    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(video_file[:-4]+'-rect.avi',
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                             fps,
                             (frame_width, frame_height))
    roi_frame = pd.read_csv(roi_file, header=[0, 1, 2], index_col=0)
    rects = roi_frame.values.astype(int)
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.namedWindow("ROI", cv2.WINDOW_KEEPRATIO)
    for i in tqdm(range(num_frame)):
        ret, frame = cap.read()
        cv2.rectangle(frame,
                      (rects[i, 0], rects[i, 1]), (rects[i, 2], rects[i, 3]),
                      (0, 255, 0), 8, cv2.LINE_AA)
        cv2.imshow("ROI", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        writer.write(frame)
    cap.release()
    writer.release()


if __name__ == '__main__':
    # generate_curve('/media/Develop/GridLight/Notebook/PER/PER20180829_6-1-Bill-2019-11-02/videos/20180829_6-1-roi-1-HRNetV2-W18-filteredSignal.h5')
    draw_roi('F:/GridLight/Data/20180829_6-1.avi',
             'F:/GridLight/Notebook/PER/PER20180829_6-1-Bill-2019-11-02/videos/20180829_6-1.csv')
