# -*- coding: utf-8 -*-
# @Time     : 2019/08/03 16:02
# @Author   : Bill H
# @Email    : lcurious@163.com
# @File     : estimate_dataset.py
# @License  : Apache-2.0
# Copyright (C) Bill H All rights reserved
import os
import random
from glob import glob

import cv2
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from natsort import natsorted
import matplotlib.pyplot as plt

from rouran.landmark_estimation.core.transform import crop


class VideoData(data.Dataset):

    def __init__(self, image_path, cfg, is_train=False, transform=None):
        # specify annotation file for dataset
        self.is_train = is_train
        self.transform = transform
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA

        # load annotations
        self.images_list = natsorted(glob(os.path.join(image_path, '*.png')))

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):

        image_path = self.images_list[idx]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        center_w = img.shape[1] / 2.0
        center_h = img.shape[0] / 2.0
        center = torch.Tensor([center_w, center_h])
        scale = max(img.shape[0], img.shape[1]) / 200.0

        img = crop(img, center, scale, self.input_size)

        img = img.astype(np.float32)
        img = (img / 255.0)
        img = img.transpose([2, 0, 1])

        meta = {'index': idx, 'center': center, 'scale': scale}

        return img, meta
