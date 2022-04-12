# -*- coding: utf-8 -*-
# @Time     : 2019/08/01 14:18
# @Author   : Bill H
# @Email    : lcurious@163.com
# @File     : landmark_dataset.py
# @License  : Apache-2.0
# Copyright (C) Bill H All rights reserved
import os
import random

import torch
import torch.utils.data as data
import pandas as pd
import math
from PIL import Image
import numpy as np

from rouran.landmark_estimation.core.transform import fliplr_joints, crop, generate_target, transform_pixel


class Per(data.Dataset):

    def __init__(self, cfg, skeletons, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET
        else:
            self.csv_file = cfg.DATASET.TESTSET

        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP
        self.matched_parts = skeletons

        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file, header=[0, 1, 2])

        # self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        # self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root,
                                  self.landmarks_frame.iloc[idx, 0])
        img = np.array(Image.open(image_path), dtype=np.float32)
        pts = self.landmarks_frame.iloc[idx, 1:].values
        pts = pts.astype('float').reshape(-1, 2)

        # xmin = np.min(pts[:, 0])
        # xmax = np.max(pts[:, 0])
        # ymin = np.min(pts[:, 1])
        # ymax = np.max(pts[:, 1])

        # center_w = (math.floor(xmin) + math.ceil(xmax)) / 2.0
        # center_h = (math.floor(ymin) + math.ceil(ymax)) / 2.0
        # pts_length = max(math.ceil(xmax) - math.floor(xmin), math.ceil(ymax) - math.floor(ymin))
        center_w = img.shape[1] / 2.0
        center_h = img.shape[0] / 2.0
        pts_length = max(img.shape[0], img.shape[1])
        scale = pts_length / 200.0
        # scale = max(img.shape[0], img.shape[1]) / 200.0
        center = torch.Tensor([center_w, center_h])

        nparts = pts.shape[0]

        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip and len(self.matched_parts):
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1],
                                    matched_parts=self.matched_parts)
                center[0] = img.shape[1] - center[0]

        img = crop(img, center, scale, self.input_size, rot=r)

        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2] + 1, center,
                                               scale, self.output_size, rot=r)
                target[i] = generate_target(target[i], tpts[i] - 1, self.sigma,
                                            label_type=self.label_type)
        img = img.astype(np.float32)
        img = (img / 255.0)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img.transpose([2, 0, 1])
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts}

        return img, target, meta
