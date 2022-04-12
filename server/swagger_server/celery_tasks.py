import os
import re
from pathlib import Path
from PIL import Image

import GPUtil
import cv2
import pims
import torch
import neotime
import pandas as pd
import tifffile

import rouran
from yacs.config import CfgNode
from swagger_server import celery_app, db_connection

# Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


@celery_app.task(name='Create Project')
def create_project(project_name, username):
    """Initialized the project content structure

    :param project_name: Project Name
    :type project_name: str
    :param username: Username
    :type username: str
    :return: If create operation OK
    :rtype: bool
    """
    workspace_path = Path(f"/workspace/{username}/{project_name}")
    samples_path = workspace_path / 'samples'
    models_path = workspace_path / 'models'
    results_path = workspace_path / 'results'
    for p in [samples_path, models_path, results_path]:
        p.mkdir(parents=True, exist_ok=True)
    return True


@celery_app.task(name='Inference Tracking')
def inference_tracking(query_node, result_node):
    """Tracking with ROI in query_node and write back results according result node

    :param query_node: neo4j graph node with query information
    :type query_node: Node
    :param result_node: neo4j graph node with result information
    :type result_node: Node
    :return:
    """
    graph, matcher = db_connection()
    sample_path = str(Path(f"/data/{result_node['sample']}"))
    result_node = matcher.match('Result', path=result_node['path']).first()

    if not Path(f"/workspace/{result_node['path']}").exists():
        Path(f"/workspace/{result_node['path']}").mkdir(parents=True, exist_ok=True)
    predicted_csv_path = str(Path(f"/workspace/{result_node['csv_path']}"))
    generated_clip_path = str(Path(f"/workspace/{result_node['clip_path']}"))
    init_boxes = {
        query_node['title']: [query_node['top_left_x'], query_node['top_left_y'],
                              query_node['bottom_right_x'], query_node['bottom_right_y']]
    }
    rouran.tracking_once(sample_path, init_boxes, predicted_csv_path)
    rouran.make_single_roi_clip(sample_path, predicted_csv_path, generated_clip_path, exist_ask=False)
    result_node.update({
        'status': 'Ready'
    })
    graph.push(result_node)
    return True


@celery_app.task(name='Create Landmark Learning Task')
def create_landmark_learning(data_node, label_node, model_node, strategy):
    """Create a training task:
    1. Extract label_file and corresponding data, then paired label and data
    2. Split labels and data in to training set and validation set
    3. Start training procedure

    :param data_node: neo4j node object with data information
    :type data_node: dict
    :param label_node: neo4j node object with label information
    :type label_node: dict
    :param model_node: neo4j node object with model information
    :type model_node: dict
    :param strategy: formed strategy for create training
    :type strategy: object
    :return:
    """
    graph, matcher = db_connection()
    # 1. Set model training directory structure
    model_path = Path(f"/workspace/{model_node['path']}")
    model_config_path = model_path / 'model_config.yaml'
    landmark_label_path = Path(f"/workspace/{label_node['path']}")
    if not landmark_label_path.exists():
        return False
    data_path = Path(f"/workspace/{data_node['clip_path']}")
    if not data_path.exists():
        return False
    landmark_df = pd.read_csv(landmark_label_path, header=[0, 1, 2], index_col=[0])
    body_parts = landmark_df.columns.get_level_values('bodyparts').unique().tolist()

    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)

    labeled_data_path = model_path / 'labeled-data'
    labeled_data_image_path = labeled_data_path / 'images'
    if not labeled_data_path.exists():
        labeled_data_path.mkdir(parents=True, exist_ok=True)
        labeled_data_image_path.mkdir(parents=True, exist_ok=True)

    new_index_list = create_datalist(str(data_path), landmark_df.index.tolist(), labeled_data_image_path)
    landmark_df.index = pd.Index(new_index_list)

    # TODO: split training set and validation set
    train_set_csv_file = labeled_data_path / 'train.csv'
    valid_set_csv_file = labeled_data_path / 'validation.csv'
    landmark_df.to_csv(str(train_set_csv_file))
    landmark_df.to_csv(str(valid_set_csv_file))

    custom_model_config = {
        'WORKERS': 0,
        'DATASET': {
            'DATASET': 'Landmark',
            'ROOT': str(labeled_data_path),
            'TRAINSET': str(train_set_csv_file),
            'TESTSET': str(valid_set_csv_file)
        },
        'MODEL': {
            'NUM_JOINTS': len(body_parts),
            'PRETRAINED': '/app/rouran/landmark_estimation/pretrained/hrnetv2_w18_imagenet_pretrained.pth'
        },
        'TRAIN': {
            'BATCH_SIZE_PER_GPU': 8
        },
        'OUTPUT_DIR': str(model_path)
    }
    custom_model_cfg = CfgNode(custom_model_config)
    rouran.landmark_estimation.default_hrnet_config.merge_from_other_cfg(custom_model_cfg)
    with open(str(model_config_path), 'w') as fp:
        print(rouran.landmark_estimation.default_hrnet_config, file=fp)

    # Update model information
    model_node = matcher.match('Model', path=model_node['path']).first()
    model_node.update({
        'status': 'Training',
        'train_set_path': str(train_set_csv_file),
        'validation_set_path': str(valid_set_csv_file),
        'config_path': str(model_config_path)
    })
    graph.push(model_node)

    # 3. start training task
    rouran.landmark_estimation.learning_landmark(str(model_config_path), max_to_keep=3)
    model_node.update({
        'status': 'Ready',
        'timestamp': str(neotime.DateTime.now()),
        'param_path': str(model_path / 'model_best.pth')
    })
    graph.push(model_node)

    return "OK"


def create_datalist(video_path, image_names, labeled_data_image_path):
    filename, file_extension = os.path.splitext(video_path)
    frame_pos_pattern = r".*frame(\d+).png"
    new_index_list = []
    if file_extension == '.tif':
        frames = tifffile.imread(video_path)
        for image_name in image_names:
            id_str = re.findall(frame_pos_pattern, image_name)[0]
            new_index_name = f"images/frame{id_str}.png"
            new_index_list.append(new_index_name)
            frame_pos = int(id_str)
            frame = frames[frame_pos]
            pim = Image.fromarray(frame)
            pim.save(f"{str(labeled_data_image_path)}/frame{id_str}.png")
    elif file_extension in ['.avi', '.mp4']:
        cap = cv2.VideoCapture(video_path)
        for image_name in image_names:
            id_str = re.findall(frame_pos_pattern, image_name)[0]
            new_index_name = f"images/frame{id_str}.png"
            new_index_list.append(new_index_name)
            frame_pos = int(id_str)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            cv2.imwrite(f"{str(labeled_data_image_path)}/frame{id_str}.png", frame)
        cap.release()
    return new_index_list


@celery_app.task(name='Inference Landmark')
def inference_landmark(result_node, model_node):
    """

    :param result_node:
    :param model_node:
    :return:
    """
    graph, matcher = db_connection()
    result_node = matcher.match('Result', path=result_node['path']).first()
    result_path = Path(f"/workspace/{result_node['path']}")
    data_path = f"/workspace/{result_node['data_path']}"
    result_csv_path = f"/workspace/{result_node['csv_path']}"
    if not result_path.exists():
        result_path.mkdir(parents=True, exist_ok=True)
    rouran.make_landmark_inference(data_path,
                                    model_node['title'], result_csv_path, model_node['config_path'])
    result_node.update({
        'status': 'Ready'
    })
    graph.push(result_node)
    # rouran.filter_predictions(project_file, [video_file], overwrite_ask=False)
    return True


@celery_app.task(name='Inference Signals')
def inference_signals(result):
    graph, matcher = db_connection()
    result_node = matcher.match('Result',title='Analysis',path=result['path']).first()
    file_path = f"/workspace/{result_node['analysis_path']}"
    tif_image = f"/data/{result_node['data_path']}"
    signal_path = f"/workspace/{result_node['csv_path']}"
    line_path =  f"/workspace/{result_node['pdf_path']}"
    video_estimations = rouran.collect_signal(file_path, tif_image, signal_path, line_path)
    result_node.update({
        'status': 'Ready'
    })
    graph.push(result_node)
    return video_estimations


@celery_app.task(name='Active Learning Selection')
def active_learning_select(project_file, video_file):
    # Get the first available GPU
    roi_video_file = str(Path(project_file).parent / 'videos' / (Path(video_file).stem + '-roi-1.avi'))
    config = rouran.utils.read_config(project_file)
    selections = rouran.extract_frames(project_file, user_feedback=False, roi_video=video_file)
    res = [int(v) for v in selections]
    return res


