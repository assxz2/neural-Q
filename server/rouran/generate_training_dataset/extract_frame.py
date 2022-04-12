# -*- coding: utf-8 -*-
# @Time     : 2019/07/16 10:26
# @Author   : Bill H
# @Email    : lcurious@163.com
# @File     : extract_frame.py
# @License  : Apache-2.0
# Copyright (C) Bill H All rights reserved
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io
from skimage.util import img_as_ubyte

from rouran import utils
from rouran.utils import frame_selection_tools


def extract_frames(config, mode='automatic', algo='kmeans', crop=False, user_feedback=True, cluster_step=1,
                   cluster_resize_width=30, cluster_color=False, opencv=True, slider_width=25, roi_video=None):
    frames2pick = []
    if mode == "manual":
        wd = Path(config).resolve().parents[0]
        os.chdir(str(wd))
        from rouran.generate_training_dataset import frame_extraction_toolbox
        frame_extraction_toolbox.show(config, slider_width)

    elif mode == "automatic":
        config_file = Path(config).resolve()
        cfg = utils.read_config(config_file)
        print("Config file read successfully.")

        numframes2pick = cfg['numframes2pick']
        start = cfg['start']
        stop = cfg['stop']

        # Check for variable correctness
        if start > 1 or stop > 1 or start < 0 or stop < 0 or start >= stop:
            raise Exception("Erroneous start or stop values. Please correct it in the config file.")
        if numframes2pick < 1 and not int(numframes2pick):
            raise Exception("Perhaps consider extracting more, or a natural number of frames.")

        videos = cfg['video_sets'].keys()
        # repeated with following rois_info
        # !!! Attention: We only handle the extracted roi videos
        roi_videos = []
        if roi_video is None:
            for vid, video in enumerate(videos):
                try:
                    roi_table_path = cfg['video_sets'][video]['rois']['file']
                    # assume the roi videos has been already extracted out to .avi video
                    # generate the roi video names according to csv description file
                    roi_table = pd.read_csv(roi_table_path, header=[0, 1, 2])
                    for scorer in roi_table.columns.get_level_values(0).unique()[1:]:
                        for roi in roi_table[scorer].columns.get_level_values(0).unique():
                            roi_video_path = os.path.splitext(roi_table_path)[0] + '-' + roi + '.avi'
                            roi_videos.append(roi_video_path)
                except Exception as e:
                    print("ROIs description file not Found!", e)
        else:
            roi_videos = [roi_video]

        import cv2
        for vindex, video in enumerate(roi_videos):
            # plt.close("all")

            if user_feedback:
                print("Do you want to extract (perhaps additional) frames for video:", video, "?")
                ask_user = input("yes/no")
            else:
                ask_user = "yes"

            if ask_user == 'y' or ask_user == 'yes':
                cap = cv2.VideoCapture(video)
                # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
                fps = cap.get(cv2.CAP_PROP_FPS)
                nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = nframes * 1. / fps
                print(nframes, fps, duration)
                index_length = int(np.ceil(np.log10(nframes)))
                if crop:
                    from rouran.utils import select_crop_parameters
                    cap.set(2, start * duration)
                    ret, frame = cap.read()
                    if ret:
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    fname = Path(video)
                    output_path = Path(config).parents[0] / 'labeled-data' / fname.stem

                    if output_path.exists():
                        fig, ax = plt.subplots(1)

                        if len(os.listdir(output_path)) == 0:  # check if empty
                            # store full frame from random location (good for augmentation)
                            index = int(start * duration + np.random.rand() * duration * (stop - start))
                            cap.set(1, index)
                            ret, frame = cap.read()
                            if ret:
                                image = img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                            img_save_path = str(output_path) + '/img' + str(index).zfill(index_length) + ".png"
                            io.imsave(img_save_path, image)

                        else:
                            ask_user = input(
                                "The directory already contains some frames. Do you want to add to it?(yes/no): ")
                            if ask_user == 'y' or ask_user == 'yes' or ask_user == 'Y' or ask_user == 'Yes':
                                index = int(start * duration + np.random.rand() * duration * (stop - start))
                                cap.set(1, index)
                                ret, frame = cap.read()
                                if ret:
                                    image = img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                                img_save_path = str(output_path) + '/img' + str(index).zfill(index_length) + ".png"
                                io.imsave(img_save_path, image)
                                pass
                            else:
                                sys.exit("Delete the frames and try again later!")

                else:
                    # without cropping a full size frame will not be extracted >>
                    # thus one more frame should be selected in next stage.
                    numframes2pick = cfg['numframes2pick'] + 1

                print("Extracting frames based on %s ..." % algo)

                if algo == 'uniform':  # extract n-1 frames (0 was already stored)
                    frames2pick = frame_selection_tools.uniform_frames_cv2(cap, numframes2pick - 1, start, stop)
                elif algo == 'kmeans':
                    frames2pick = frame_selection_tools.Kmeans_based_frame_selection_cv2(cap, numframes2pick - 1, start,
                                                                                         stop,
                                                                                         step=cluster_step,
                                                                                         resizewidth=cluster_resize_width,
                                                                                         color=cluster_color)
                else:
                    print("Please implement this method yourself and send us a pull request! "
                          "Otherwise, choose 'uniform' or 'kmeans'.")
                    frames2pick = []

                video_name = Path(video).stem
                name_span = re.search(r'-roi-[0-9]', video_name).span()
                output_path = Path(config).parents[0] / 'labeled-data' / video_name[:name_span[0]] / \
                              video_name[name_span[0] + 1:name_span[1]]
                for index in frames2pick:
                    cap.set(1, index)  # extract a particular frame
                    ret, frame = cap.read()
                    if ret:
                        image = img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        img_name = '{0}/img{1}.png'.format(str(output_path), str(index).zfill(index_length))
                        io.imsave(img_name, image)
                    else:
                        print("Frame", index, " not found!")
                cap.release()
    else:
        print("Invalid MODE. Choose either 'manual' or 'automatic'. "
              "Check ``help(rouran.extract_frames)`` on python and ``rouran.extract_frames?`` "
              "for ipython/jupyter notebook for more details.")

    print("\nFrames were selected."
          "\nYou can now label the frames using the function 'label_frames' "
          "(if you extracted enough frames for all videos).")
    return frames2pick


if __name__ == '__main__':
    extract_frames('../../examples/Tracing-Bill-2019-07-15/config.yaml')
