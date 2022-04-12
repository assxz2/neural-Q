# -*- coding: utf-8 -*-
# @Time     : 2019/08/01 11:01
# @Author   : Huang Zenan
# @Email    : lcurious@163.com
# @File     : create_trial.py
# @License  : Apache-2.0
# Copyright (C) Huang Zenan All rights reserved
import os
from pathlib import Path
from rouran.landmark_estimation.train import train
from rouran.utils import utils, create_logger


logger = create_logger("rouran.create_trial")


def train_network(config, shuffle=1, training_set_index=0, gpu_to_use=None, max_snapshots_to_keep=5, auto_tune=False,
                  display_iters=10, save_iters=10, max_iters=None):
    """Trains the network with the labels in the training dataset.

    """
    # reload logger.

    start_path = os.getcwd()

    # Read file path for pose_config file. >> pass it on
    cfg = utils.read_config(config)
    model_folder_name = utils.get_model_folder(cfg["TrainingFraction"][training_set_index], shuffle, cfg)
    hrnet_config_file = Path(os.path.join(cfg['project_path'],
                                          str(model_folder_name), "train", "landmark_config.yaml"))
    if not hrnet_config_file.is_file():
        logger.info("The training datafile ", hrnet_config_file, " is not present.")
        logger.info("Probably, the training dataset for this secific shuffle index was not created.")
        logger.info("Try with a different shuffle/training_set_fraction or use function 'create_training_dataset' "
                    "to create a new training dataset with this shuffle index.")
    else:
        try:
            train(cfg, display_iters, save_iters,
                  max_to_keep=max_snapshots_to_keep)  # pass on path and file name for pose_cfg.yaml!
            pass
        except BaseException as e:
            raise e
        finally:
            os.chdir(str(start_path))
        logger.info("The network is now trained and ready to evaluate. "
                    "Use the function 'evaluate_network' to evaluate the network.")
