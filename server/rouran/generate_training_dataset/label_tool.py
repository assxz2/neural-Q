# -*- coding: utf-8 -*-
# @Time     : 2019/08/09 21:54
# @Author   : Bill H
# @Email    : lcurious@163.com
# @File     : frame_selection_tools.py
# @License  : Apache-2.0
# Copyright (C) Bill H All rights reserved
import os
import re
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.widgets import Button, RadioButtons
from natsort import natsorted

from rouran.utils import utils

ANNOTATOR = None


class AnnotationWin(object):
    def __init__(self, data_path, config):
        roi_name = str(Path(data_path).stem)
        name = str(Path(data_path).parent.stem)
        self.scorer = config['scorer']
        self.dotsize = config['dotsize']
        self.data_path = data_path
        self.img_ptr = 0
        self.fig, self.ax = plt.subplots(num="Annotating-[{} of {}]".format(roi_name, name))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.subplots_adjust(bottom=0.2)
        self.joint_names = config['bodyparts']
        self.image_files = natsorted(glob(os.path.join(data_path, '*.png')))
        self.relative_image_paths = [Path(im_path).parts[-4] + '/' +
                                     Path(im_path).parts[-3] + '/' +
                                     Path(im_path).parts[-2] + '/' +
                                     Path(im_path).parts[-1]
                                     for im_path in self.image_files]
        self.save_file_name = str(Path(self.data_path) / 'CollectedData_') + self.scorer
        self.table_multi_index = pd.MultiIndex.from_product(
            [[config['scorer']],
             config['bodyparts'],
             ['x', 'y']],
            names=['scorer', 'bodyparts', 'coords'])
        self.data_frame = None
        # Merge data from exists dataframe
        try:
            self.data_frame = pd.read_hdf(self.save_file_name + '.h5')
            old_images = sorted(list(self.data_frame.index))
            self.new_image_paths = list(set(self.relative_image_paths) - set(old_images))
            if len(self.new_image_paths):
                table = np.empty((len(self.new_image_paths), len(config['bodyparts']) * 2))
                table[:] = np.nan
                new_data_frame = pd.DataFrame(table, index=self.new_image_paths, columns=self.table_multi_index)
                self.data_frame = pd.concat([new_data_frame, self.data_frame])
                self.data_frame.sort_index(inplace=True)
        except Exception as e:
            print("=> New dataframe create!")
            # create labels table
            table = np.empty((len(self.relative_image_paths), len(config['bodyparts']) * 2))
            table[:] = np.nan
            self.data_frame = pd.DataFrame(table, columns=self.table_multi_index, index=self.relative_image_paths)

        # show first frame
        self.img = np.asarray(Image.open(self.image_files[0]))
        self.ax.imshow(self.img)
        self.scale_ratio = self.fig.dpi*self.fig.get_size_inches()[1]/(self.img.shape[0])
        print(self.fig.dpi*self.fig.get_size_inches(), self.img.shape)

        # Button position
        self.ax_prev = plt.axes([0.7, 0.05, 0.08, 0.06])
        self.ax_next = plt.axes([0.8, 0.05, 0.08, 0.06])
        self.ax_save = plt.axes([0.6, 0.05, 0.08, 0.06])
        self.ax_delete = plt.axes([0.5, 0.05, 0.08, 0.06])

        # Two process control button
        self.btn_next = Button(self.ax_next, 'Next')
        self.btn_next.on_clicked(self.on_press_next)
        self.btn_prev = Button(self.ax_prev, 'Prev')
        self.btn_prev.on_clicked(self.on_press_prev)
        self.btn_save = Button(self.ax_save, 'Save')
        self.btn_save.on_clicked(self.save)
        self.btn_delete = Button(self.ax_delete, 'Del')
        self.btn_delete.on_clicked(self.delete)
        # Point plot connection
        self.fig.canvas.mpl_connect('button_press_event', self.on_press_image)
        # Pick point connection
        self.fig.canvas.mpl_connect('pick_event', self.on_pick_points)
        self.move_point_cid = self.fig.canvas.mpl_connect('button_press_event', self.move_point)
        self.picked_point = None
        self.allow_move_point = False
        self.picked_point_id = None
        # title information
        self.ax.title.set_text("{} / {}".format(self.img_ptr + 1, len(self.image_files)))
        # Radio button
        self.radio_ax = plt.axes([0.05, 0.2, 0.15, 0.02 * len(self.joint_names)])
        self.btn_radio = RadioButtons(self.radio_ax, self.joint_names)
        # initial with read table file
        self.point_ptr = self.redraw_with_labels()
        self.btn_radio.set_active(self.point_ptr % len(self.joint_names))

    def redraw_with_labels(self):
        self.ax.cla()
        self.img = np.asarray(Image.open(self.image_files[self.img_ptr]))
        self.ax.imshow(self.img)
        point_ptr = 0
        image_name = self.relative_image_paths[self.img_ptr]
        image_coords = self.data_frame.loc[image_name]
        for jn in self.joint_names:
            x = image_coords[self.scorer, jn, 'x']
            y = image_coords[self.scorer, jn, 'y']
            if not np.isnan(x) and not np.isnan(y):
                self.ax.plot(x, y, 'or', alpha=0.5, picker=5,
                             markersize=self.dotsize*self.scale_ratio)
                point_ptr += 1
        self.btn_radio.set_active(point_ptr % len(self.joint_names))
        return point_ptr

    def on_press_next(self, event):
        self.added_points = []
        self.point_ptr = 0
        if self.img_ptr < len(self.image_files) - 1:
            self.img_ptr += 1
            self.point_ptr = self.redraw_with_labels()
            self.ax.title.set_text("{} / {}".format(self.img_ptr + 1, len(self.relative_image_paths)))

    def on_press_prev(self, event):
        self.added_points = []
        self.point_ptr = 0
        if self.img_ptr > 0:
            self.img_ptr -= 1
            self.point_ptr = self.redraw_with_labels()
            self.ax.title.set_text("{} / {}".format(self.img_ptr + 1, len(self.relative_image_paths)))

    def on_press_image(self, event):
        if event.inaxes == self.ax and event.button == 3:
            if self.point_ptr < len(self.joint_names):
                point, = self.ax.plot(event.xdata, event.ydata, 'or', alpha=0.5, picker=5,
                                      markersize=self.dotsize*self.scale_ratio)
                # write label into file
                index = self.relative_image_paths[self.img_ptr]
                bp_name = self.joint_names[self.point_ptr]
                self.data_frame.loc[index][self.scorer, bp_name, 'x'] = event.xdata
                self.data_frame.loc[index][self.scorer, bp_name, 'y'] = event.ydata
                # Update point number
                self.point_ptr += 1
                self.btn_radio.set_active(self.point_ptr)
            else:
                self.ax.title.set_text("{} / {} \nAll points added, you can drag one".format(
                    self.img_ptr + 1, len(self.relative_image_paths)))

    def on_pick_points(self, event):
        points = event.artist
        ind = event.ind
        self.allow_move_point = True
        # may multiple picked at once
        if isinstance(ind, list):
            self.picked_point = points[0]
            self.picked_point_id = self.get_point_id(str(self.picked_point))
        else:
            self.picked_point = points
            self.picked_point_id = self.get_point_id(str(self.picked_point))
        jn = self.joint_names[self.picked_point_id]
        self.ax.title.set_text("joint {} picked".format(jn))

    def get_point_id(self, point_name):
        num_str = re.findall(r'[(]_line(\d+)[)]', point_name)[0]
        return int(num_str)

    def move_point(self, event):
        if self.picked_point is not None:
            self.picked_point.set_data([event.xdata], [event.ydata])
            if event.button == 3:
                self.allow_move_point = False
                self.picked_point = None
                # write result back into table
                image_name = self.relative_image_paths[self.img_ptr]
                jn = self.joint_names[self.picked_point_id]
                self.data_frame.loc[image_name][self.scorer, jn, 'x'] = event.xdata
                self.data_frame.loc[image_name][self.scorer, jn, 'y'] = event.ydata

    def delete(self, event):
        image_name = self.relative_image_paths[self.img_ptr]
        image_coords = self.data_frame.loc[image_name]
        if self.point_ptr > 0:
            self.point_ptr -= 1
            # self.point_ptr point to next label to label
            # so point_ptr - 1 point to last labeled label
            image_coords[self.scorer, self.joint_names[self.point_ptr], 'x'] = np.nan
            image_coords[self.scorer, self.joint_names[self.point_ptr], 'y'] = np.nan
            self.redraw_with_labels()

    def save(self, event):
        # for human read
        self.data_frame.to_csv(self.save_file_name + '.csv')
        print("=> ", self.save_file_name + '.csv Saved!')
        # for machine read
        self.data_frame.to_hdf(self.save_file_name + '.h5', 'df_with_missing', format='table', mode='w')
        print("=> ", self.save_file_name + '.h5 Saved!')


def label_frames(config_file):
    global ANNOTATOR
    config = utils.read_config(config_file)
    project_path = config['project_path']
    data_path = Path(project_path) / 'labeled-data'
    videos = config['video_sets'].keys()
    for video_path in videos:
        video_name = Path(video_path).stem
        video_label_path = Path(data_path) / video_name
        rois = os.listdir(str(video_label_path))
        for roi in rois:
            roi_label_path = Path(video_label_path) / roi
            print(roi_label_path)
            print("=> Start annotate {}-{}".format(video_name, roi))
            if len(glob(os.path.join(roi_label_path, '*.png'))):
                ANNOTATOR = AnnotationWin(roi_label_path, config)
            else:
                print("Nothing to annotate in ", roi_label_path)
