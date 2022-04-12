# -*- coding: utf-8 -*-
# @Time     : 2019/08/06 15:19
# @Author   : Huang Zenan
# @Email    : lcurious@163.com
# @File     : evaluate.py
# @License  : Apache-2.0
# Copyright (C) Huang Zenan All rights reserved
from pathlib import Path

import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from yacs.config import CfgNode

from rouran.landmark_estimation.core import function
from rouran.landmark_estimation.core.evaluation import decode_preds, compute_nme
from rouran.landmark_estimation.dataset import Per
from rouran.landmark_estimation.models import get_landmark_alignment_net
from rouran.utils import utils, create_logger
import torch


logger = create_logger("rouran.evaluate")


def evaluate_network(config_file, shuffles=[1], gpu_to_use=[0], show_errors=True):
    cfg = utils.read_config(config_file)
    training_set_folder = utils.get_training_set_folder(cfg)
    # make evaluation workspace for validation
    utils.attempt_to_make_folder(Path(cfg['project_path']) / 'evaluation-results')
    for shuffle in shuffles:
        for train_fraction in cfg['TrainingFraction']:
            model_folder = Path(cfg['project_path']) / utils.get_model_folder(train_fraction, shuffle, cfg)
            test_config_path = Path(model_folder) / 'test' / 'landmark_config.yaml'
            with open(test_config_path) as fs:
                test_hrnet_cfg = CfgNode.load_cfg(fs)
            model_path = Path(test_hrnet_cfg.OUTPUT_DIR) / 'model_best.pth'
            # Load complete model from best model file
            model = get_landmark_alignment_net(test_hrnet_cfg)
            try:
                model = torch.load(model_path)
                model = nn.DataParallel(model, device_ids=gpu_to_use).cuda()
                logger.info("=> loaded best model best_model.pth")
            except FileExistsError:
                logger.info("=> no best model found")
            model.eval()

            eval_loader = DataLoader(
                dataset=Per(test_hrnet_cfg, False),
                batch_size=test_hrnet_cfg.TEST.BATCH_SIZE_PER_GPU,
                num_workers=test_hrnet_cfg.WORKERS,
                pin_memory=test_hrnet_cfg.PIN_MEMORY
            )
            nme, predictions = function.inference(test_hrnet_cfg, eval_loader, model)
            logger.info("=> Evaluation finished, NME is ", nme)
