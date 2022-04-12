# -*- coding: utf-8 -*-
# @Time     : 2019/07/15 14:31
# @Author   : Huang Zenan
# @Email    : lcurious@163.com
# @File     : create.py
# @License  : Apache-2.0
# Copyright (C) Huang Zenan All rights reserved
import os
import shutil
from pathlib import Path
import pandas as pd

import cv2

from rouran.utils import utils, create_logger
from datetime import datetime

logger = create_logger("rouran")


def create_new_workspace(workspace, experimenter, videos, working_directory=None, copy_video=False, video_type='.avi'):
    """
    Create a new workspace and subdirectory for whole pipeline in processing experiment data, configure the default
    parameters latter
    :param workspace: string, string containing the name of workspace
    :param experimenter: sting, name of creator username
    :param videos: list, list of video url
    :param working_directory: string, path located in cloud server
    :param copy_video: bool, if copy video to workspace path
    :param video_type: string
    :return:
    """
    exp_date = datetime.today()
    exp_date = exp_date.strftime('%Y-%m-%d-%H%M%S')
    if working_directory is None:
        working_directory = '.'
    wd_path = Path(working_directory).resolve()
    workspace_name = '{wn}-{exp}-{date}'.format(wn=workspace, exp=experimenter, date=exp_date)
    workspace_path = wd_path / workspace_name

    # Create workspace and sub-directories structure
    if workspace_path.exists():
        logger.info('Workspace "{}" already exists!'.format(workspace_path))
        return os.path.join(str(workspace_path), 'config.yaml')

    video_path = workspace_path / 'videos'
    data_path = workspace_path / 'labeled-data'
    shuffles_path = workspace_path / 'training-datasets'
    results_path = workspace_path / 'models'
    transforms_path = workspace_path / 'transforms'
    for p in [video_path, data_path, shuffles_path, results_path, transforms_path]:
        p.mkdir(parents=True, exist_ok=True)
        logger.info('Created "{}"'.format(p))

    # Import all videos in a folder or if just one video without [] passed, then make it a list
    if isinstance(videos, str):
        # pending file or list
        if os.path.isdir(videos):
            path = videos
            # todo: video extension check
            videos = [os.path.join(path, vp) for vp in os.listdir(path) if video_path in vp]
            if len(videos) == 0:
                logger.error("No videos found in", path, os.listdir(path))
                logger.error("Perhaps change the video_type, which is currently set:", video_type)
            else:
                logger.info("Directory entered, ", len(videos), "video were found.")
        else:
            if os.path.isfile(videos):
                videos = [videos]

    videos = [Path(vp) for vp in videos]
    dirs = [data_path / Path(i.stem) for i in videos]
    for p in dirs:
        # Create directory under data
        p.mkdir(parents=True, exist_ok=True)

    destinations = [video_path.joinpath(vp.name) for vp in videos]
    if copy_video:
        logger.info("Copying the videos")
        for src, dst in zip(videos, destinations):
            shutil.copy(os.fspath(src), os.fspath(dst))
            src_csv = os.path.splitext(src)[0] + '.csv'
            dst_csv = os.path.splitext(dst)[0] + '.csv'
            try:
                shutil.copy(os.fspath(src_csv), os.fspath(dst_csv))
            except FileNotFoundError:
                logger.error("ROIs description {} not found!".format(src_csv))
    else:
        logger.info("Creating the symbolic link of the video")
        for src, dst in zip(videos, destinations):
            if dst.exists():
                raise FileExistsError('Video {} exists already!'.format(dst))
            try:
                src = str(src)
                dst = str(dst)
                os.symlink(src, dst)
                src_csv = os.path.splitext(src)[0] + '.csv'
                dst_csv = os.path.splitext(dst)[0] + '.csv'
                try:
                    shutil.copy(os.fspath(src_csv), os.fspath(dst_csv))
                except FileNotFoundError:
                    logger.error("ROIs description {} not found!".format(src_csv))
            except OSError:
                import subprocess
                subprocess.check_call('mklink {s} {d}'.format(s=src, d=dst), shell=True)
            logger.info('Created the symlink of {} to {}'.format(src, dst))
            videos = destinations

    if copy_video:
        videos = destinations

    video_sets = {}
    for video in videos:
        try:
            rel_video_path = str(Path.resolve(Path(video)))
        except:
            rel_video_path = os.readlink(str(video))

        cap = cv2.VideoCapture(rel_video_path)
        ret, frame = cap.read()
        # if roi tracking file exited merge into this project
        roi_dict = {'file': os.path.splitext(video)[0] + '.csv'}
        if Path(roi_dict['file']).exists():
            roi_table = pd.read_csv(roi_dict['file'], header=[0, 1, 2], index_col=[0])
            roi_names = roi_table.columns.get_level_values(1).unique()
            for ir, roi_n in enumerate(roi_names):
                roi_dict[roi_n] = roi_table.iloc[0].values[ir * 4:ir * 4 + 4].tolist()
        else:
            roi_dict['roi-1'] = []
        if cap.isOpened():
            video_sets[rel_video_path] = {
                # rois should be project specific
                'rois': roi_dict,
            }
        else:
            print("Cannot open the video file")
            video_sets = None

    # Set up the config.yaml
    cfg_file, ruamel_file = utils.create_config_template()
    cfg_file['Task'] = workspace
    cfg_file['scorer'] = experimenter
    cfg_file['video_sets'] = video_sets
    cfg_file['project_path'] = str(workspace_path)
    cfg_file['transforms_path'] = str(transforms_path)
    cfg_file['date'] = exp_date
    cfg_file['bodyparts'] = ['A8-L', 'A8-R', 'A7-L', 'A7-R', 'A6-L', 'A6-R', 'A5-L', 'A5-R',
                             'A4-L', 'A4-R', 'A3-L', 'A3-R', 'A2-L', 'A2-R', 'A1-L', 'A1-R']
    cfg_file['cropping'] = False
    cfg_file['start'] = 0
    cfg_file['stop'] = 1
    cfg_file['numframes2pick'] = 20
    cfg_file['TrainingFraction'] = [0.95]
    cfg_file['iteration'] = 0
    cfg_file['net'] = 'hrnetv2_w18'
    cfg_file['snapshotindex'] = -1
    cfg_file['x1'] = 0
    cfg_file['x2'] = 640
    cfg_file['y1'] = 277
    cfg_file['y2'] = 624
    cfg_file['batch_size'] = 8  # batch size during inference (video - analysis);
    cfg_file['corner2move2'] = (50, 50)
    cfg_file['move2corner'] = True
    cfg_file['skeleton'] = [['A8-L', 'A8-R'], ['A7-L', 'A7-R'], ['A6-L', 'A6-R'], ['A5-L', 'A5-R'],
                            ['A4-L', 'A4-R'], ['A3-L', 'A3-R'], ['A2-L', 'A2-R'], ['A1-L', 'A1-R']]
    cfg_file['skeleton_color'] = 'black'
    cfg_file['pcutoff'] = 0.1
    cfg_file['dotsize'] = 12  # for plots size of dots
    cfg_file['alphavalue'] = 0.7  # for plots transparency of markers
    cfg_file['colormap'] = 'jet'  # for plots type of colormap

    # write out to yaml config file
    workspace_config_file = os.path.join(str(workspace_path), 'config.yaml')
    utils.write_config(workspace_config_file, cfg_file)
    logger.info('Generated "{}"'.format(workspace_path / 'config.yaml'))
    logger.info("\n"
                "A new project with name %s is created at %s and a configurable file (config.yaml) is stored there. \n"
                "Change the parameters in this file to adapt to your project's needs.\n "
                "Once you have changed the configuration file, "
                "use the function 'extract_frames' to select frames for labeling.\n"
                "[OPTIONAL] Use the function 'add_new_videos' to add new videos to your project (at any stage)." % (
                    workspace_name, str(wd_path)))
    return workspace_config_file


if __name__ == '__main__':
    from rouran.create_workspace import add_new_videos

    add_new_videos('/media/Develop/GridLight/rouran/examples/Tracing-Bill-2019-07-16/config.yaml',
                   ['/media/Develop/GridLight/Data/20180829_4-1.avi'], copy_videos=True)
