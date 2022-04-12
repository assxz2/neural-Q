# -*- coding: utf-8 -*-
# @Time     : 2019/07/16 19:28
# @Author   : Huang Zenan
# @Email    : lcurious@163.com
# @File     : add.py
# @License  : Apache-2.0
# Copyright (C) Huang Zenan All rights reserved
from rouran.utils import create_logger

logger = create_logger("rouran.add")


def add_new_videos(config, videos, copy_videos=False, coords=None):
    import os
    import shutil
    from pathlib import Path

    from rouran.utils import utils
    import cv2

    # Read the config file
    cfg = utils.read_config(config)

    video_path = Path(config).parents[0] / 'videos'
    data_path = Path(config).parents[0] / 'labeled-data'
    videos = [Path(vp) for vp in videos]

    dirs = [data_path / Path(i.stem) for i in videos]

    for p in dirs:
        """
        Creates directory under data & perhaps copies videos (to /video)
        """
        p.mkdir(parents=True, exist_ok=True)

    destinations = [video_path.joinpath(vp.name) for vp in videos]
    if copy_videos == True:
        for src, dst in zip(videos, destinations):
            if dst.exists():
                pass
            else:
                logger.info("Copying the videos")
                shutil.copy(os.fspath(src), os.fspath(dst))
                src_csv = os.path.splitext(src)[0] + '.csv'
                dst_csv = os.path.splitext(dst)[0] + '.csv'
                try:
                    shutil.copy(os.fspath(src_csv), os.fspath(dst_csv))
                except FileNotFoundError:
                    logger.error("ROIs description {} not found!".format(src_csv))
    else:
        for src, dst in zip(videos, destinations):
            if dst.exists():
                pass
            else:
                logger.info("Creating the symbolic link of the video")
                src = str(src)
                dst = str(dst)
                os.symlink(src, dst)
                src_csv = os.path.splitext(src)[0] + '.csv'
                dst_csv = os.path.splitext(dst)[0] + '.csv'
                try:
                    shutil.copy(os.fspath(src_csv), os.fspath(dst_csv))
                except FileNotFoundError:
                    logger.info("ROIs description {} not found!".format(src_csv))

    if copy_videos == True:
        videos = destinations  # in this case the *new* location should be added to the config file
    # adds the video list to the config.yaml file
    for idx, video in enumerate(videos):
        try:
            # For windows os.path.realpath does not work and does not link to the real video.
            video_path = str(Path.resolve(Path(video)))
        #           video_path = os.path.realpath(video)
        except:
            video_path = os.readlink(video)

        vcap = cv2.VideoCapture(video_path)
        if vcap.isOpened():
            # get vcap property
            width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if coords == None:
                cfg['video_sets'].update({video_path: {'crop': ', '.join(map(str, [0, width, 0, height]))}})
            else:
                c = coords[idx]
                cfg['video_sets'].update({video_path: {'crop': ', '.join(map(str, c))}})
        else:
            logger.error("Cannot open the video file!")

    utils.write_config(config, cfg)
    logger.info("New video was added to the project! Use the function 'extract_frames' to select frames for labeling.")
