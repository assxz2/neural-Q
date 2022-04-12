# -*- coding: utf-8 -*-
# @Time     : 2019/08/14 00:56
# @Author   : Bill H
# @Email    : lcurious@163.com
# @File     : train.py
# @License  : Apache-2.0
# Copyright (C) Bill H All rights reserved
import rouran
import argparse
import sys


def parse_args():
    """
    args for onekey.
    """
    parser = argparse.ArgumentParser(description='Train rouran network for test.')
    # for train
    parser.add_argument('--cfg', type=str, help='yaml configure file name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config_file_path = args.cfg
    # check crop
    # rouran.crop_videos(config_file_path)
    # check extracted
    # rouran.extract_frames(config_file_path)
    # check labeled
    # check create train set
    # rouran.create_training_dataset(config_file_path)
    # start training
    rouran.train_network(config_file_path)


if __name__ == '__main__':
    main()
