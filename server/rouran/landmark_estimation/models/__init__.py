# -*- coding: utf-8 -*-
# @Time     : 2019/08/01 10:33
# @Author   : Bill H
# @Email    : lcurious@163.com
# @File     : __init__.py.py
# @License  : Apache-2.0
# Copyright (C) Bill H All rights reserved

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng (tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .hrnet import get_landmark_alignment_net, HighResolutionNet

__all__ = ['HighResolutionNet', 'get_landmark_alignment_net']
