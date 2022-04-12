# -*- coding: utf-8 -*-
# @Time     : 2019/07/15 16:27
# @Author   : Huang Zenan
# @Email    : lcurious@163.com
# @File     : utils.py
# @License  : Apache-2.0
# Copyright (C) Huang Zenan All rights reserved
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import re
import logging
from pathlib import Path

import ruamel.yaml
import yaml

import rouran


def create_logger(logger_name):
    """
    Create logger for whole project

    :param logger_name: string
    :return: class, logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # create handler output to file
    handler = logging.FileHandler("{n}.log".format(n=logger_name), mode='a')
    handler.setLevel(logging.INFO)

    # create handler output to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # setup logging format
    log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(log_formatter)
    console_handler.setFormatter(log_formatter)

    logger.addHandler(handler)
    logger.addHandler(console_handler)
    return logger


def create_config_template():
    ruamel_file = ruamel.yaml.YAML()
    # based on the project directory
    parent_path = Path(os.path.dirname(rouran.__file__))
    default_config_file = str(parent_path / 'constants/template_config.yaml')
    with open(default_config_file) as fp:
        cfg_file = ruamel_file.load(fp)
    return cfg_file, ruamel_file


def read_config(config_name):
    """
    Reads structured config file
    :param config_name:
    :return:
    """
    ruamel_file = ruamel.yaml.YAML()
    path = Path(config_name)
    if os.path.exists(path):
        try:
            with open(path, 'r') as fp:
                cfg = ruamel_file.load(fp)
        except Exception as err:
            if err.args[2] == "could not determine a constructor for the tag '!!python/tuple'":
                with open(path, 'r') as yaml_file:
                    cfg = yaml.load(yaml_file, Loader=yaml.SafeLoader)
                    write_config(config_name, cfg)
    else:
        raise FileNotFoundError(
            "Config file is not found. Please make sure that the file exists"
            " and/or there are no unnecessary spaces in the path of the config file!")
    return cfg


def write_config(config_name, cfg):
    """
    Write structured config file
    :param config_name:
    :param cfg:
    :return:
    """
    with open(config_name, 'w') as cf:
        cfg_file, ruamel_file = create_config_template()
        for key in cfg.keys():
            cfg_file[key] = cfg[key]

        # Adding default value for variable skeleton and skeleton_color for backward compatibility
        if 'skeleton' not in cfg.keys():
            cfg_file['skeleton'] = []
            cfg_file['skeleton_color'] = 'black'
        ruamel_file.dump(cfg_file, cf)


def attempt_to_make_folder(folder_name, recursive=False):
    """
    Attempts to create a folder with specified name. Does nothing if it already exists.
    :param folder_name:
    :param recursive:
    :return: is_existed
    """

    try:
        os.path.isdir(folder_name)
    except TypeError:
        # https://www.python.org/dev/peps/pep-0519/
        folder_name = os.fspath(folder_name)

    if os.path.isdir(folder_name):
        print(folder_name, " already exists!")
        is_existed = True
    else:
        if recursive:
            os.makedirs(folder_name)
        else:
            os.mkdir(folder_name)
        is_existed = False
    return is_existed


# Read the pickle file
def read_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)


# Write the pickle file
def write_pickle(filename, data):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_list_of_videos(videos, video_type):
    from random import sample
    # checks if input is a directory
    if [os.path.isdir(i) for i in videos] == [True]:  # os.path.isdir(video)==True:
        """
        Analyzes all the videos in the directory.
        """

        print("Analyzing all the videos in the directory")
        # todo: use re here
        # pattern r'*roi-d*labeled.video_type'
        pattern = re.compile(r'.*-roi-\d.*' + str(video_type))
        video_folder = videos[0]
        os.chdir(video_folder)
        video_list = [fn for fn in os.listdir(os.curdir) if
                      (pattern.match(fn) is not None) and ('labeled.mp4' not in fn)]  # exclude labeled-videos!

        videos = sample(video_list,
                        len(video_list))  # this is useful so multiple nets can be used to analzye simultanously
    else:
        if isinstance(videos, str):
            if os.path.isfile(videos):  # #or just one direct path!
                videos = [v for v in videos if os.path.isfile(v) and ('labeled.mp4' not in v)]
            else:
                videos = []
        else:
            videos = [v for v in videos if os.path.isfile(v) and ('labeled.mp4' not in v)]
    return videos


def get_model_folder(train_fraction, shuffle, cfg):
    task = cfg['Task']
    date = cfg['date']
    iterate = 'iteration-' + str(cfg['iteration'])
    return Path('models/' + iterate + '/' + task + '-' + date + '-TrainSet-' +
                str(int(train_fraction * 100)) + '-Shuffle' + str(shuffle))


def get_model_name(config):
    if config['net'] == 'hrnetv2_w18':
        return 'HRNetV2-W18'


# Various functions to get filenames, folder names etc. based on configuration parameters.
def get_training_set_folder(cfg):
    ''' Training Set folder for config file based on parameters '''
    Task = cfg['Task']
    date = cfg['date']
    iterate = 'iteration-' + str(cfg['iteration'])
    return Path(os.path.join('training-datasets', iterate, 'UnaugmentedDataSet_' + Task + '_' + date))


def get_evaluation_folder(train_fraction, shuffle, cfg):
    Task = cfg['Task']
    date = cfg['date']
    iterate = 'iteration-' + str(cfg['iteration'])
    return Path('evaluation-results/' + iterate + '/' + Task + date + '-trainset' + str(
        int(train_fraction * 100)) + 'shuffle' + str(shuffle))
