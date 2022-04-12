# -*- coding: utf-8 -*-
# @Time     : 2019/08/01 20:25
# @Author   : Bill H
# @Email    : lcurious@163.com
# @File     : create_training.py
# @License  : Apache-2.0
# Copyright (C) Bill H All rights reserved
import os

import numpy as np
import pandas as pd
from yacs.config import CfgNode as CN

from rouran.landmark_estimation import model_utils, default_hrnet_config
from rouran.utils import utils, Path


def make_train_pose_yaml(items_to_change, save_as_config_file):
    items_change_cfg = CN(items_to_change)
    default_hrnet_config.merge_from_other_cfg(items_change_cfg)
    with open(save_as_config_file, 'w') as f:
        print(default_hrnet_config, file=f)
    return default_hrnet_config


def make_test_pose_yaml(train_config, items_to_change, save_as_file):
    items_change_cfg = CN(items_to_change)
    train_config.merge_from_other_cfg(items_change_cfg)
    with open(save_as_file, "w") as f:
        print(train_config, file=f)
    return train_config


def merge_annotated_datasets(cfg, project_path, training_set_folder_path):
    """
    Merges all the h5 files for all labeled-datasets (from individual videos).
    This is a bit of a mess because of cross platform compatablity.

    Please Use Ubuntu
    """
    annotation_data = None
    data_path = Path(os.path.join(project_path, 'labeled-data'))
    videos = cfg['video_sets'].keys()
    for v_name in videos:
        vn = Path(v_name).stem
        try:
            roi_table = pd.read_csv(cfg['video_sets'][v_name]['rois']['file'], header=[0, 1, 2], index_col=[0])
            for scorer in roi_table.columns.get_level_values(0).unique():
                for roi in roi_table[scorer].columns.get_level_values(0).unique():
                    try:
                        data = pd.read_csv(str(data_path / Path(vn) / roi) + '/CollectedData_' + cfg['scorer'] + '.csv')
                        if annotation_data is None:
                            annotation_data = data
                        else:
                            annotation_data = pd.concat([annotation_data, data])
                    except FileNotFoundError:
                        print((str(data_path / Path(vn) / roi) + '/CollectedData_' + cfg['scorer'] + '.csv'),
                              " not found (perhaps not annotated)")
        except FileNotFoundError:
            print(os.path.splitext(v_name)[0] + '.csv',
                  " not found (perhaps not annotated)")

    # store to training step input file
    file_name = str(str(training_set_folder_path) + '/CollectedData_' + cfg['scorer'])
    # For machine read
    annotation_data.to_hdf(file_name + '.h5', key='df_with_missing', mode='w')
    # For Human read
    annotation_data.to_csv(file_name + '.csv', index=False)
    return annotation_data, file_name + '.csv'


def split_trials(trial_index, train_fraction=0.8):
    ''' Split a trial index into train and test sets. Also checks that the trainFraction is a two digit number between 0 an 1. The reason
    is that the folders contain the trainfraction as int(100*trainFraction). '''
    if train_fraction > 1 or train_fraction < 0:
        print(
            "The training fraction should be a two digit number between 0 and 1; i.e. 0.95. Please change accordingly.")
        return [], []

    if abs(train_fraction - round(train_fraction, 2)) > 0:
        print(
            "The training fraction should be a two digit number between 0 and 1; i.e. 0.95. Please change accordingly.")
        return [], []
    else:
        train_set_size = int(len(trial_index) * round(train_fraction, 2))
        shuffle = np.random.permutation(trial_index)
        test_indexes = shuffle[train_set_size:]
        train_indexes = shuffle[:train_set_size]

        return train_indexes, test_indexes


def create_training_dataset(config_file, num_shuffles=1, shuffles=None, train_indexes=None,
                            test_indexes=None):
    """
    Creates a training dataset. Labels from all the extracted frames are merged into a single .h5 file.\n
    Only the videos included in the config file are used to create this dataset.\n

    [OPTIONAL] Use the function 'add_new_video' at any stage of the project to add more videos to the project.

    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.

    num_shuffles : int, optional
        Number of shuffles of training dataset to create, i.e. [1,2,3] for num_shuffles=3. Default is set to 1.

    shuffles: list of shuffles.
        Alternatively the user can also give a list of shuffles (integers!).

    windows2linux: bool.
        The annotation files contain path formated according to your operating system. If you label on windows
        but train & evaluate on a unix system (e.g. ubunt, colab, Mac) set this variable to True to convert the paths.

    Example
    --------
    """

    # Loading metadata from config file:
    cfg = utils.read_config(config_file)
    scorer = cfg['scorer']
    project_path = cfg['project_path']
    # Create path for training sets & store data there
    training_set_folder = utils.get_training_set_folder(cfg)  # Path concatenation OS platform independent
    utils.attempt_to_make_folder(Path(os.path.join(project_path, str(training_set_folder))), recursive=True)

    # Load annotation data
    data, data_labels_path = merge_annotated_datasets(cfg, project_path,
                                                      Path(os.path.join(project_path, training_set_folder)))
    # this score labeled data
    data = data[scorer]

    # load pretrained network parameters
    net_type = cfg['net']
    import rouran
    parent_path = Path(os.path.dirname(rouran.__file__))

    model_path, num_shuffles = model_utils.find_pretrained(net_type, parent_path, num_shuffles)

    if shuffles == None:
        shuffles = range(1, num_shuffles + 1, 1)
    else:
        shuffles = [i for i in shuffles if isinstance(i, int)]

    body_parts = cfg['bodyparts']
    train_fraction_set = cfg['TrainingFraction']
    for shuffle in shuffles:
        for training_fraction in train_fraction_set:
            if train_indexes is None and test_indexes is None:
                train_indexes, test_indexes = split_trials(range(len(data.index)), training_fraction)
            else:
                print("You passed a split with the following fraction:",
                      len(train_indexes) * 1. / (len(test_indexes) + len(train_indexes)) * 100)

            model_folder_name = utils.get_model_folder(training_fraction, shuffle, cfg)
            utils.attempt_to_make_folder(Path(config_file).parents[0] / model_folder_name, recursive=True)
            utils.attempt_to_make_folder(str(Path(config_file).parents[0] / model_folder_name) + '/train')
            utils.attempt_to_make_folder(str(Path(config_file).parents[0] / model_folder_name) + '/test')

            train_config_path = str(Path(cfg['project_path']) / model_folder_name / 'train/landmark_config.yaml')
            test_config_path = str(Path(cfg['project_path']) / model_folder_name / 'test/landmark_config.yaml')

            items_to_change = {
                'DATASET': {
                    'DATASET': 'Landmark',
                    'ROOT': cfg['project_path'],
                    'TRAINSET': data_labels_path,
                    'TESTSET': data_labels_path
                },
                'MODEL': {
                    'NUM_JOINTS': len(body_parts),
                    'PRETRAINED': model_path,
                },
                'TRAIN': {
                  'BATCH_SIZE_PER_GPU': cfg['batch_size']
                },
                'OUTPUT_DIR': str(Path(train_config_path).parents[0])
            }
            train_config = make_train_pose_yaml(items_to_change, train_config_path)
            make_test_pose_yaml(train_config, items_to_change, test_config_path)
