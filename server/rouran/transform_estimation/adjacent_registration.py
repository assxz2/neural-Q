# -*- coding: utf-8 -*-
# @Time     : 2019/07/15 22:31
# @Author   : Bill H
# @Email    : lcurious@163.com
# @File     : adjacent_registration.py
# @License  : Apache-2.0
# Copyright (C) Bill H All rights reserved
import glob
import multiprocessing as mp
import os
from pathlib import Path

import cv2
import numpy as np
import ants
import pandas as pd
import skimage.io as skio
from skimage import img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.transform import resize
from rouran import utils
from tqdm import tqdm


def denoise_image(noisy_image):
    patch_kw = dict(patch_size=5,  # 5x5 patches
                    patch_distance=6,  # 13x13 search area
                    multichannel=True)
    image = img_as_float(noisy_image)
    sigma_est = np.mean(estimate_sigma(image, multichannel=True))
    denoise_image = denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=False,
                                     **patch_kw)
    return denoise_image


def pair_registration(fix_id, mov_id, info_list):
    """
    make a estimation of registration
    :param fix_id:
    :param mov_id:
    :return:
    """
    print("Start process {fi} <=> {mi} registration".format(fi=fix_id, mi=mov_id))
    fix_path = sample_paths[fix_id]
    mov_path = sample_paths[mov_id]
    fix_image_id = os.path.split(fix_path)[-1].split('-')[0]
    mov_image_id = os.path.split(mov_path)[-1].split('-')[0]
    tx_path_full = os.path.join(tx_path, '{fi}-{mi}'.format(fi=fix_image_id,
                                                            mi=mov_image_id))
    if not os.path.exists(tx_path_full):
        os.mkdir(tx_path_full)
    # elif len(os.listdir(tx_path_full)) == 3:
    #     return tx_path_full

    fix_image = skio.imread(fix_path)
    fix_image = resize(fix_image,
                       (fix_image.shape[0] * 4, fix_image.shape[1] * 4),
                       anti_aliasing_sigma=None,
                       anti_aliasing=False,
                       preserve_range=True,
                       mode='edge',
                       order=0)
    mov_image = skio.imread(mov_path)
    mov_image = resize(mov_image,
                       (mov_image.shape[0] * 4, mov_image.shape[1] * 4),
                       anti_aliasing_sigma=None,
                       anti_aliasing=False,
                       preserve_range=True,
                       mode='edge',
                       order=0)

    # registration part
    fixed = ants.from_numpy(fix_image)
    moving = ants.from_numpy(mov_image)

    tx = ants.registration(fixed=fixed, moving=moving,
                           type_of_transform='SyNRA',
                           syn_metric='CC',
                           syn_sampling=128,
                           outprefix=tx_path_full + '/')

    metric_fwd = ants.image_similarity(tx['warpedmovout'], fixed,
                                       metric_type='MeanSquares')
    metric_inv = ants.image_similarity(tx['warpedfixout'], moving,
                                       metric_type='MeanSquares')
    msg = {
        'imageA': fix_path,
        'imageB': mov_path,
        'tx_fwd': tx['fwdtransforms'],
        'tx_inv': tx['invtransforms'],
        'metric_fwd': metric_fwd,
        'metric_inv': metric_inv
    }
    info_list.append(msg)
    print("Finish process {fi} <=> {mi} registration".format(fi=fix_id, mi=mov_id))
    return tx_path_full


def write_info(info_list):
    """
    write the registration message into info table
    :param info_list:
    :return:
    """
    csv_path = tx_path + '.csv'
    print("Start write information to: ", csv_path)
    table_dict = {
        'imageA': [],
        'imageB': [],
        'tx_fwd': [],
        'tx_inv': [],
        'metric_fwd': [],
        'metric_inv': []
    }
    for item in info_list:
        print("forward metric: ", item['metric_fwd'], " & inverse metric: ", item['metric_inv'])
        table_dict['imageA'].append(item['imageA'])
        table_dict['imageB'].append(item['imageB'])
        table_dict['tx_fwd'].append(item['tx_fwd'])
        table_dict['tx_inv'].append(item['tx_inv'])
        table_dict['metric_fwd'].append(item['metric_fwd'])
        table_dict['metric_inv'].append(item['metric_inv'])
    data_frame = pd.DataFrame.from_dict(table_dict)
    data_frame.to_csv(tx_path + '.csv')


def run():
    print("Start execute tasks!")
    for i in range(sample_num - 1):
        task_pool.apply_async(pair_registration, (i, i + 1, info_list,))
    task_pool.close()
    task_pool.join()
    print("All task executed!")


def task():
    pass


def status():
    """
    return the information of current stage
    :return:
    """
    table = []
    return table


def estimate_transforms(config_file):
    """
    estimate all transforms of adjacent frames of video
    :param config_file:
    :return:
    """
    config = utils.read_config(config_file)
    video_sets = config['video_sets']
    # make the roi label path
    project_path = Path(config['project_path'])
    transforms_path = project_path / 'transforms'
    for key in video_sets.keys():
        video_roi_table_path = video_sets[key]['rois']['file']
        print("Estimating transforms according to : ", video_roi_table_path)
        roi_table = pd.read_csv(video_roi_table_path, header=[0, 1, 2])
        video_transforms_path = transforms_path / Path(key).stem
        utils.attempt_to_make_folder(video_transforms_path)
        for scorer in roi_table.columns.get_level_values(0).unique()[1:]:
            for roi in roi_table[scorer].columns.get_level_values(0).unique():
                utils.attempt_to_make_folder(video_transforms_path / roi)
                roi_video = os.path.splitext(key)[0] + '-' + roi + '.avi'
                video_cap = cv2.VideoCapture(roi_video)
                antspy_registration(video_cap, video_transforms_path / roi)


def antspy_registration(cap, output_folder):
    """
    registration with ANTsPy
    make previous frame the moving image, waited image as fixed image
    :param cap:
    :param output_folder: output folder with full path
    :return:
    """
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = Path(output_folder)
    output_info_path = output_path / 'transforms.xlsx'
    ret, first_frame = cap.read()
    print(first_frame.shape)
    prev_frame = denoise_image(first_frame)
    moving = ants.from_numpy(prev_frame)
    registration_info = {
        'tx_fwd': [],
        'tx_inv': [],
        'metric_fwd': [],
        'metric_inv': []
    }
    for i in tqdm(range(1, nframes)):
        ret, frame = cap.read()
        clean_frame = denoise_image(frame)
        fixed = ants.from_numpy(clean_frame)
        # print(str(output_path) + '/{:0>4d}'.format(i))
        tx = ants.registration(fixed=fixed, moving=moving,
                               type_of_transform='SyN',
                               syn_metric='CC',
                               syn_sampling=64,
                               outprefix=str(output_path) + '/{:0>4d}'.format(i))
        metric_fwd = ants.image_similarity(tx['warpedmovout'], fixed,
                                           metric_type='MeanSquares')
        metric_inv = ants.image_similarity(tx['warpedfixout'], moving,
                                           metric_type='MeanSquares')
        registration_info['tx_fwd'].append(tx['fwdtransforms'])
        registration_info['tx_inv'].append(tx['invtransforms'])
        registration_info['metric_fwd'].append(metric_fwd)
        registration_info['metric_inv'].append(metric_inv)
        # next estimation the moving image should be updated
        moving = fixed
    registration_info_table = pd.DataFrame(registration_info)
    registration_info_table.to_excel(output_info_path)
    print("Transform estimation done, information file saved to:\n", output_info_path)


if __name__ == '__main__':
    # data_path = '/media/ubuntu/fdh/Data/LightSheet/6-1-roi'
    # # data_path = '/mnt/Develop/Projects/DiffeomorphicCF/cache/6-1-roi'
    # sample_paths = natsorted(glob.glob(os.path.join(data_path, '*.png')))
    # sample_num = len(sample_paths)
    # tx_path = data_path + 'x4-T'
    # if not os.path.exists(tx_path):
    #     os.mkdir(tx_path)
    # # transformations number
    # tx_num = 0
    # # registration status
    # info_list = mp.Manager().list()
    # status_table = []
    # prefix = ''
    # task_pool = mp.Pool(processes=8)
    # run()
    # write_info(info_list)
    estimate_transforms('../../examples/Tracing-Bill-2019-07-16/config.yaml')
