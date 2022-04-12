# -*- coding: utf-8 -*-
# @Time     : 2019/08/03 15:43
# @Author   : Huang Zenan
# @Email    : lcurious@163.com
# @File     : estimate.py
# @License  : Apache-2.0
# Copyright (C) Huang Zenan All rights reserved
import os
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import io
import base64

import pims
from IPython.display import HTML
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from yacs.config import CfgNode as CN
import tifffile

from rouran.landmark_estimation.core.evaluation import decode_preds
from rouran.landmark_estimation.dataset.estimate_dataset import VideoData
from rouran.utils import utils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from .models import get_landmark_alignment_net


def make_estimation(videos, config_file):
    cfg = utils.read_config(config_file)
    model_folder_name = utils.get_model_folder(cfg["TrainingFraction"][0], 1, cfg)
    hrnet_config_file = Path(os.path.join(cfg['project_path'],
                                          str(model_folder_name), "train", "landmark_config.yaml"))
    with open(hrnet_config_file) as f:
        hrnet_cfg = CN.load_cfg(f)

    num_classes = hrnet_cfg.MODEL.NUM_JOINTS
    video_list = []
    if isinstance(videos, str):
        video_list.append(videos)
    else:
        video_list = videos

    # load model
    model = get_landmark_alignment_net(hrnet_cfg)
    model_state_file = os.path.join(hrnet_cfg.OUTPUT_DIR, 'model_best.pth')
    print(model_state_file)
    if os.path.isfile(model_state_file):
        model = torch.load(model_state_file)
        # model.load_state_dict(best_state_dict)
        model = nn.DataParallel(model, device_ids=[0]).cuda()
        print("=> loaded checkpoint best_model.pth")
    else:
        print("=> no checkpoint found")
    model.eval()

    estimated_csv_list = []
    for video_file in video_list:
        v_cap = cv2.VideoCapture(video_file)
        # make temp video folder
        tmp_video_path = Path(video_file).parents[0] / Path(video_file).stem
        if not Path(tmp_video_path).exists():
            Path.mkdir(tmp_video_path)
            nframes = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            index_length = int(np.ceil(np.log10(nframes)))
            for fid in tqdm(range(nframes)):
                ret, frame = v_cap.read()
                cv2.imwrite(os.path.join(tmp_video_path, str(fid).zfill(index_length)+'.png'), frame)

        v_data_loader = DataLoader(
            dataset=VideoData(tmp_video_path, hrnet_cfg),
            batch_size=hrnet_cfg.TEST.BATCH_SIZE_PER_GPU,
            num_workers=hrnet_cfg.WORKERS,
            pin_memory=hrnet_cfg.PIN_MEMORY,
        )

        predictions = torch.zeros((len(v_data_loader.dataset), num_classes, 2))
        with torch.no_grad():
            try:
                for i, (inputs, meta) in enumerate(tqdm(v_data_loader)):
                    output = model(inputs)
                    score_map = output.data.cpu()
                    preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
                    for n in range(score_map.size(0)):
                        predictions[meta['index'][n], :, :] = preds[n, :, :]
            except Exception as e:
                print("Something wrong!", e)

        model_name = utils.get_model_name(cfg)
        pred_numpy = predictions.numpy()
        pred_dataframe = pd.DataFrame(data=pred_numpy.reshape([-1, num_classes * 2]),
                                      columns=pd.MultiIndex.from_product(
                                          [[str(model_name)], cfg["bodyparts"], ["x", "y"]],
                                          names=['model_name', 'joint_name', 'coords']))
        # merge labeled results into the estimation results
        pred_dataframe_name = str(tmp_video_path) + '-' + str(model_name)
        print(pred_dataframe_name)
        # drop unrecognized values
        pred_dataframe[pred_dataframe < 0] = None
        # pred_dataframe.interpolate(method='linear', axis=1, inplace=True)
        pred_dataframe.ffill(axis=1, inplace=True)
        pred_dataframe.bfill(axis=1, inplace=True)
        pred_dataframe.to_csv(pred_dataframe_name + '.csv')
        pred_dataframe.to_csv(str(tmp_video_path) + '_keypoints.csv')
        estimated_csv_list.append(str(tmp_video_path) + '_keypoints.csv')
        pred_dataframe.to_hdf(pred_dataframe_name + '.h5', 'prediction')
    # clear cuda memory after estimation done!
    torch.cuda.empty_cache()
    return estimated_csv_list


def make_landmark_inference(video_file, model_name, predicted_landmark_file, hrnet_config_file):
    with open(hrnet_config_file) as f:
        hrnet_cfg = CN.load_cfg(f)

    num_classes = hrnet_cfg.MODEL.NUM_JOINTS

    # load model
    model = get_landmark_alignment_net(hrnet_cfg)
    model_state_file = os.path.join(hrnet_cfg.OUTPUT_DIR, 'model_best.pth')
    print(model_state_file)
    if os.path.isfile(model_state_file):
        model = torch.load(model_state_file)
        # model.load_state_dict(best_state_dict)
        model = nn.DataParallel(model, device_ids=[0]).cuda()
        print("=> loaded checkpoint best_model.pth")
    else:
        print("=> no checkpoint found")
    model.eval()

    filename, file_extension = os.path.splitext(video_file)

    if file_extension in ['.avi', '.mp4']:
        v_cap = cv2.VideoCapture(video_file)
        # print("Total Frame: ", v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # make temp video folder
        tmp_video_path = Path(video_file).parents[0] / Path(video_file).stem
        if not Path(tmp_video_path).exists():
            Path.mkdir(tmp_video_path)
            nframes = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            index_length = int(np.ceil(np.log10(nframes)))
            for fid in tqdm(range(nframes)):
                ret, frame = v_cap.read()
                cv2.imwrite(os.path.join(tmp_video_path, str(fid).zfill(index_length)+'.png'), frame)
    elif file_extension == '.tif':
        frames = tifffile.imread(video_file)
        tmp_video_path = Path(video_file).parents[0] / Path(video_file).stem
        if not Path(tmp_video_path).exists():
            Path.mkdir(tmp_video_path)
            nframes = len(frames)
            index_length = int(np.ceil(np.log10(nframes)))
            for fid in tqdm(range(nframes)):
                frame = frames[fid]
                pim = Image.fromarray(frame)
                pim.save(os.path.join(tmp_video_path, str(fid).zfill(index_length)+'.png'))

    v_data_loader = DataLoader(
        dataset=VideoData(tmp_video_path, hrnet_cfg),
        batch_size=hrnet_cfg.TEST.BATCH_SIZE_PER_GPU,
        num_workers=0,
        pin_memory=hrnet_cfg.PIN_MEMORY,
    )

    predictions = torch.zeros((len(v_data_loader.dataset), num_classes, 2))
    with torch.no_grad():
        try:
            for i, (inputs, meta) in enumerate(tqdm(v_data_loader)):
                output = model(inputs)
                score_map = output.data.cpu()
                preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
                for n in range(score_map.size(0)):
                    predictions[meta['index'][n], :, :] = preds[n, :, :]
        except Exception as e:
            print("Something wrong!", e)

    pred_numpy = predictions.numpy()
    label_dataframe = pd.read_csv(hrnet_cfg.DATASET.TESTSET, header=[0, 1, 2], index_col=[0])
    pred_dataframe = pd.DataFrame(data=pred_numpy.reshape([-1, num_classes * 2]),
                                  columns=label_dataframe.columns)
    # merge labeled results into the estimation results
    pred_dataframe_name = str(tmp_video_path) + '-' + str(model_name)
    print(pred_dataframe_name)
    # drop unrecognized values
    pred_dataframe[pred_dataframe < 0] = None
    # pred_dataframe.interpolate(method='linear', axis=1, inplace=True)
    pred_dataframe.ffill(axis=1, inplace=True)
    pred_dataframe.bfill(axis=1, inplace=True)
    pred_dataframe.to_csv(predicted_landmark_file)
    # clear cuda memory after estimation done!
    torch.cuda.empty_cache()
    return predicted_landmark_file


def create_labeled_video(config_file, videos):
    cfg = utils.read_config(config_file)
    video_list = []
    model_name = utils.get_model_name(cfg)
    colors = plt.cm.get_cmap(cfg['colormap'], len(cfg['bodyparts']))
    if isinstance(videos, str):
        video_list.append(videos)
    else:
        video_list = videos
    for video_file in video_list:
        estimate_file = str(Path(video_file).parent / Path(video_file).stem) + '-' + model_name + '-filtered' + '.h5'
        if Path(estimate_file).exists():
            estimate_table = pd.read_hdf(estimate_file)
            v_cap = cv2.VideoCapture(video_file)
            nframes = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            nframes_digits = int(np.ceil(np.log10(nframes)))
            frame_width = v_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = v_cap.get(cv2.CAP_PROP_FPS)
            labeled_video_path = estimate_file[:-3]
            if not Path(labeled_video_path).exists():
                Path(labeled_video_path).mkdir()
            labeled_video_file = labeled_video_path + '-labeled.mp4'
            if Path(labeled_video_file).exists():
                ans = input("Labeled Video already generate, do you want to overwrite it? (yes/[no])")
                if ans == 'yes' or ans == 'y':
                    pass
                else:
                    video = io.open(labeled_video_file,
                                    'r+b').read()
                    encoded = base64.b64encode(video)
                    return HTML(data='''<video alt="test" controls style="clear:both;display:block;margin:auto">
                                        <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                                        </video>'''.format(encoded.decode('ascii')))
            for i in tqdm(range(nframes)):
                ret, image = v_cap.read()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_name = labeled_video_path + "/file" + str(i).zfill(nframes_digits) + ".png"
                fig = plt.figure(frameon=False, figsize=(
                    frame_width * 2. / 64, frame_height * 2. / 64), dpi=64)
                bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                width, height = bbox.width * fig.dpi, bbox.height * fig.dpi
                scale_ratio = height / image.shape[0]
                plt.subplots_adjust(left=0, bottom=0, right=1,
                                    top=1, wspace=0, hspace=0)
                plt.imshow(image)

                coords = estimate_table.loc[i][model_name]
                for bpi, bp in enumerate(cfg['bodyparts']):
                    x = coords[bp, 'x']
                    y = coords[bp, 'y']
                    plt.scatter([x], [y], s=(cfg['dotsize']*2*scale_ratio)**2,
                                c=[colors(bpi)], alpha=cfg['alphavalue'] * .75)
                plt.xlim(0, frame_width-1)
                plt.ylim(0, frame_height-1)

                plt.axis('off')
                plt.subplots_adjust(
                    left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                plt.gca().invert_yaxis()
                # plt.tight_layout()
                plt.savefig(image_name, dpi=64)
                plt.close("all")

            # merge images to video file
            start = os.getcwd()
            os.chdir(labeled_video_path)
            print("All labeled frames were created, now generating video...")
            # One can change the parameters of the video creation script below:
            # See ffmpeg user guide: http://ffmpeg.org/ffmpeg.html#Video-and-Audio-file-format-conversion
            #
            print("Labeled video [{}] generating...".format(labeled_video_file))
            try:
                ffmpeg_instruction = [
                    'ffmpeg', '-framerate',
                    str(fps), '-i', 'file%0' + str(nframes_digits) + 'd.png',
                    '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                    '-r', str(fps),
                    labeled_video_file, '-y']
                # print(' '.join(ffmpeg_instruction))
                subprocess.call(ffmpeg_instruction)
                print("Labeled video [{}] generated!".format(labeled_video_file))

            except FileNotFoundError:
                print("ffmpeg not correctly installed.")
            os.chdir(start)
            video = io.open(labeled_video_file,
                            'r+b').read()
            encoded = base64.b64encode(video)
            return HTML(data='''<video alt="test" controls style="clear:both;display:block;margin:auto">
                                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                                </video>'''.format(encoded.decode('ascii')))
        else:
            print("Post processing not run, please use rouran.filter_predictions(config_file_path, extract_video)")
