# -*- coding: utf-8 -*-
# @Time     : 2019/08/01 13:51
# @Author   : Huang Zenan
# @Email    : lcurious@163.com
# @File     : train.py
# @License  : Apache-2.0
# Copyright (C) Huang Zenan All rights reserved
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from rouran.landmark_estimation import model_utils
from rouran.landmark_estimation import models
from rouran.landmark_estimation.core import function
from rouran.landmark_estimation.dataset import get_dataset
from rouran.utils import utils
from .hrnet_config import _C as default_hrnet_config


def train(cfg, display_iters=10, save_iters=5, max_to_keep=5, training_set_index=0, shuffle=1):
    start_path = os.getcwd()
    # switch to folder where store config.yaml
    model_folder_name = utils.get_model_folder(cfg["TrainingFraction"][training_set_index], shuffle, cfg)
    config_file = str(Path(os.path.join(cfg['project_path'],
                                        str(model_folder_name), "train", "landmark_config.yaml")))
    os.chdir(str(Path(config_file).parents[0]))
    config = default_hrnet_config.clone()
    config.merge_from_file(config_file)
    model_logger, final_output_dir, tb_log_dir = model_utils.create_logger(config, 'train')

    model_logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = models.get_landmark_alignment_net(config)

    # copy model files
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    device = torch.device(gpus[0])
    model = nn.DataParallel(model, device_ids=gpus).cuda(device)

    # loss
    criterion = torch.nn.MSELoss(reduction='mean').cuda(device)

    optimizer = model_utils.get_optimizer(config, model)
    best_nme = 100
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'latest.pth')
        if os.path.islink(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_nme = checkpoint['best_nme']
            model = checkpoint['state_dict']
            optimizer.load_state_dict(checkpoint['optimizer'])
            model_logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            model_logger.warn("=> no checkpoint found")

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    dataset_type = get_dataset(config)

    b_i_map = dict()
    for b_id, b_name in enumerate(cfg['bodyparts']):
        b_i_map[b_name] = b_id
    matched_parts = []
    # for item in cfg['skeleton']:
    #     matched_parts.append([b_i_map[item[0]], b_i_map[item[1]]])
    train_loader = DataLoader(
        dataset=dataset_type(config,
                             matched_parts,
                             is_train=True),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=0,
        pin_memory=config.PIN_MEMORY)

    val_loader = DataLoader(
        dataset=dataset_type(config,
                             matched_parts,
                             is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=0,
        pin_memory=config.PIN_MEMORY
    )

    for epoch in tqdm(range(last_epoch, config.TRAIN.END_EPOCH)):
        function.train_epoch(config, train_loader, model, criterion,
                             optimizer, epoch, writer_dict)
        # After 1.1.0 changed this behavior in a BC-breaking way.
        lr_scheduler.step(epoch)

        # evaluate
        nme, predictions = function.validate(config, val_loader, model,
                                             criterion, epoch, writer_dict)

        is_best = nme < best_nme
        best_nme = min(nme, best_nme)

        model_logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        # DEBUG
        # print("best:", is_best)
        if epoch % save_iters == 0:
            model_utils.save_checkpoint(
                {"state_dict": model,
                 "epoch": epoch + 1,
                 "best_nme": best_nme,
                 "optimizer": optimizer.state_dict(),
                 }, predictions, is_best, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth')
    model_logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


def learning_landmark(config_file, display_iters=10, save_iters=5, max_to_keep=5, training_set_index=0, shuffle=1):
    start_path = os.getcwd()
    # switch to folder where store config.yaml
    os.chdir(str(Path(config_file).parents[0]))
    config = default_hrnet_config.clone()
    config.merge_from_file(config_file)
    model_logger, final_output_dir, tb_log_dir = model_utils.create_logger(config, 'train')

    model_logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = models.get_landmark_alignment_net(config)

    # copy model files
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    device = torch.device(gpus[0])
    model = nn.DataParallel(model, device_ids=gpus).cuda(device)

    # loss
    criterion = torch.nn.MSELoss(reduction='mean').cuda(device)

    optimizer = model_utils.get_optimizer(config, model)
    best_nme = 100
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'latest.pth')
        if os.path.islink(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_nme = checkpoint['best_nme']
            model = checkpoint['state_dict']
            optimizer.load_state_dict(checkpoint['optimizer'])
            model_logger.info("=> loaded checkpoint (epoch {})"
                              .format(checkpoint['epoch']))
        else:
            model_logger.warn("=> no checkpoint found")

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    dataset_type = get_dataset(config)

    matched_parts = []
    # for item in cfg['skeleton']:
    #     matched_parts.append([b_i_map[item[0]], b_i_map[item[1]]])
    train_loader = DataLoader(
        dataset=dataset_type(config,
                             matched_parts,
                             is_train=True),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=0,
        pin_memory=config.PIN_MEMORY)

    val_loader = DataLoader(
        dataset=dataset_type(config,
                             matched_parts,
                             is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=0,
        pin_memory=config.PIN_MEMORY
    )

    for epoch in tqdm(range(last_epoch, config.TRAIN.END_EPOCH)):
        function.train_epoch(config, train_loader, model, criterion,
                             optimizer, epoch, writer_dict)
        # After 1.1.0 changed this behavior in a BC-breaking way.
        lr_scheduler.step(epoch)

        # evaluate
        nme, predictions = function.validate(config, val_loader, model,
                                             criterion, epoch, writer_dict)

        is_best = nme < best_nme
        best_nme = min(nme, best_nme)

        model_logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        # DEBUG
        # print("best:", is_best)
        if epoch % save_iters == 0:
            model_utils.save_checkpoint(
                {"state_dict": model,
                 "epoch": epoch + 1,
                 "best_nme": best_nme,
                 "optimizer": optimizer.state_dict(),
                 }, predictions, is_best, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth')
    model_logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()
