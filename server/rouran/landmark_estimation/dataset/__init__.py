# -*- coding: utf-8 -*-
# @Time     : 2019/08/01 14:15
# @Author   : Bill H
# @Email    : lcurious@163.com
# @File     : __init__.py.py
# @License  : Apache-2.0
# Copyright (C) Bill H All rights reserved
from .landmark_dataset import Per


def get_dataset(config):

    if config.DATASET.DATASET == 'Landmark':
        return Per
    else:
        raise NotImplemented()
