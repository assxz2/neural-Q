# -*- coding: utf-8 -*-
# @Time     : 2019/08/02 10:41
# @Author   : Bill H
# @Email    : lcurious@163.com
# @File     : model_utils.py
# @License  : Apache-2.0
# Copyright (C) Bill H All rights reserved
import logging
import os
import time

import cv2
import torch
from torch import optim
import numpy as np

from rouran.utils import Path


def find_pretrained(model_type, parent_path, num_shuffles):
    """
    gets local path to network weights and checks if they are present. If not, raise Error!
    :param model_type:
    :param parent_path:
    :param num_shuffles:
    :return:
    """
    if model_type == 'hrnetv2_w18':
        model_path = parent_path / 'landmark_estimation/pretrained/hrnetv2_w18_imagenet_pretrained.pth'
    else:
        print("Currently only hrnetv2_18 supported!")
        num_shuffles = -1
        model_path = parent_path

    if num_shuffles > 0:
        if not model_path.is_file():
            raise FileNotFoundError("hrnetv2_w18_imagenet_pretrained.pth not found!")

    return str(model_path), num_shuffles


# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn), Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------
def create_logger(config, phase='train'):
    # set up logger
    root_output_dir = Path(config.OUTPUT_DIR)
    if not root_output_dir.exists():
        print('=> model folder not exist! do you finish training set generation?')

    dataset = config.DATASET.DATASET
    model = config.MODEL.NAME

    final_output_dir = root_output_dir

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(phase, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        filemode='w',
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(config.LOG_DIR) / (phase + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer


def save_checkpoint(states, predictions, is_best,
                    output_dir, filename='checkpoint.pth'):
    preds = predictions.cpu().data.numpy()
    torch.save(states, os.path.join(output_dir, filename))
    torch.save(preds, os.path.join(output_dir, 'current_pred.pth'))

    latest_path = os.path.join(output_dir, 'latest.pth')
    if os.path.islink(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.join(output_dir, filename), latest_path)

    if is_best and 'state_dict' in states.keys():
        torch.save(states['state_dict'].module, os.path.join(output_dir, 'model_best.pth'))


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_numpy(img):
    if torch.is_tensor(img):
        img = to_numpy(img)
        img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


def im_to_torch(img):
    if not torch.is_tensor(img):
        img = np.transpose(img, (2, 0, 1))  # C*H*W
        img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def gauss(x, a, b, c, d=0):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2)) + d


def color_heatmap(x):
    x = to_numpy(x)
    color = np.zeros((x.shape[0], x.shape[1], 3))
    color[:, :, 0] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
    color[:, :, 1] = gauss(x, 1, .5, .3)
    color[:, :, 2] = gauss(x, 1, .2, .3)
    color[color > 1] = 1
    color = (color * 255).astype(np.uint8)
    return color


def sample_with_heatmap(inp, out, num_rows=2, parts_to_show=None):
    inp = to_numpy(inp * 255)
    out = to_numpy(out)

    img = np.zeros((inp.shape[1], inp.shape[2], inp.shape[0]))
    for i in range(3):
        img[:, :, i] = inp[i, :, :]

    if parts_to_show is None:
        parts_to_show = np.arange(out.shape[0])

    num_cols = int(np.ceil(float(len(parts_to_show)) / num_rows))
    size = img.shape[0] // num_rows

    full_img = np.zeros((img.shape[0], size * (num_cols + num_rows), 3), np.uint8)
    full_img[:img.shape[0], :img.shape[1]] = img

    inp_small = cv2.resize(img, (size, size))

    # Set up heatmap display for each part
    for i, part in enumerate(parts_to_show):
        part_idx = part
        out_resized = cv2.resize(out[part_idx], (size, size))
        out_resized = out_resized.astype(float)
        out_img = inp_small.copy() * .3
        color_hm = color_heatmap(out_resized)
        out_img += color_hm * .7

        col_offset = (i % num_cols + num_rows) * size
        row_offset = (i // num_cols) * size
        full_img[row_offset:row_offset + size, col_offset:col_offset + size] = out_img

    return full_img


def batch_with_heatmap(inputs, outputs, mean=torch.Tensor([0.5, 0.5, 0.5]), num_rows=2, parts_to_show=None):
    batch_img = []
    for n in range(min(inputs.size(0), 2)):
        inp = inputs[n] + mean.view(3, 1, 1).expand_as(inputs[n])
        batch_img.append(
            sample_with_heatmap(inp.clamp(0, 1), outputs[n], num_rows=num_rows, parts_to_show=parts_to_show)
        )
    return im_to_torch(np.concatenate(batch_img))
