# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
from math import floor

import multiprocessing as mp
import pickle
import numpy as np
import os
import tempfile

import warnings
import cv2
import tqdm
import logging

import yaml

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from analysis import PossessionAnalysis
from analysis import TerritoryAnalysis
from scene_classification import SceneClass

from predictor import VisualizationDemo
from field_registation import SportsFieldRegistrator
from jersey_colour_detection import JerseyColourDetection
from scene_classification import SceneClassification

# constants
WINDOW_NAME = "COCO detections"
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def setup_cfg(args):
    # load config from file and command-line arguments
    cfgs = []
    for model in args: 
        cfg = get_cfg()
        model = args[model]
        # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
        # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
        # add_panoptic_deeplab_config(cfg)
        cfg.merge_from_file(model["config"])
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = model["conf_threshold"] # Set threshold for this model
        cfg.MODEL.WEIGHTS = model["weights"] # Set path model .pth
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = model["num_classes"]
        cfg.DATASETS.TEST = tuple(model["class_names"])
        cfg.freeze()
        cfgs.append(cfg)
    return cfgs


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="config.yaml",
        metavar="FILE",
        help="path to config file",
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

def log_to_file(scene, previous_scene, possession, territory, i): 
    if previous_scene != scene:
        logging.info(f"Transitioned to {scene} from {previous_scene} at {floor(i/(frames_per_second*60))}m{floor(i/frames_per_second)}s. Current Possession: {possession.possession_stats()*100}. Current Territory: {territory.teamA_territory(weighted=False)}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    logging.basicConfig(filename='test.log', format='%(levelname)s:%(message)s', level=logging.INFO)
    with open(args.config_file, "r") as yamlfile: 
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)

    cfg = setup_cfg(data["detectron2"])

    demo = VisualizationDemo(cfg)
    edge_map_simulation_ranges = {
        "pan_range" : [-52, 50, 1], 
        "tilt_range" : [-250, 400, 25], 
        "zoom_range" : [1, 2, 1]
    }

    field_registor = SportsFieldRegistrator(data["field_registration"]["frame"], data["field_registration"]["model"], data["field_registration"]["simulation_ranges"], data["field_registration"]["line_mask"], data["video"]["verbose"])
    field_registor.generateSimulatedEdgeMaps(data["field_registration"]["keypoints"])
    jersey_detector = JerseyColourDetection(
        data["jersey_detection"]["teamA_colour"], 
        data["jersey_detection"]["teamB_colour"], 
        data["jersey_detection"]["offical_colour"], 
    )
    possession_analysis = PossessionAnalysis(data["jersey_detection"]["teamA_colour"][2])
    territory_analysis = TerritoryAnalysis()

    court_model = cv2.imread(data["field_registration"]["model"])

    scene_classification = SceneClassification([(33, 0, 0), (70, 255, 255)], 0.5, 0.02)
    corner_points = np.zeros((1,4))

    assert os.path.isfile(data["video"]["input"])
    video = cv2.VideoCapture(data["video"]["input"])
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    basename = os.path.basename(data["video"]["input"])
    codec, file_ext = (
        ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
    )
    if codec == ".mp4v":
        warnings.warn("x264 codec not available, switching to mp4v")



    j = 0
    previous_scene = None
    key_frame = cv2.imread(data["field_registration"]["frame"])

    for vis_frame, predictions in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
        player_detections = predictions[0]
        ball_detections = predictions[1]

        scene_type = scene_classification.frame_classification(vis_frame, [vis_frame[y_bottom:y_top, x_bottom:x_top] for x_bottom, y_bottom, x_top, y_top in player_detections.pred_boxes.tensor.numpy().astype(int)])
        if scene_type == SceneClass.NON_FIELD: 
            continue

        tranformation_matrix, canny_frame = field_registor.pitch_registration(vis_frame, np.array([[x_bottom, y_bottom, x_top, y_top] for x_bottom, y_bottom, x_top, y_top in player_detections.pred_boxes.tensor.numpy()]), scene_type, corner_points)
        inv_tranformation_matrix = np.linalg.inv(tranformation_matrix)

        center_bottom_prediction = np.array([[[x_top + (x_bottom - x_top)/2, y_bottom] for x_top, y_top, x_bottom, y_bottom in player_detections.pred_boxes.tensor.numpy()]])
        vis_frame_corners = np.array([[(0, vis_frame.shape[0]), (0, 0), (vis_frame.shape[1], 0), (vis_frame.shape[1], vis_frame.shape[0])]]) * [key_frame.shape[1] / vis_frame.shape[1], key_frame.shape[0] / vis_frame.shape[0]]
        corner_points = cv2.perspectiveTransform(vis_frame_corners.astype(float), inv_tranformation_matrix, (court_model.shape[1], court_model.shape[0]))

        if center_bottom_prediction.size == 0: 
            transformed_points = np.array([[]])
        else: 
            transformed_points = cv2.perspectiveTransform(center_bottom_prediction * [key_frame.shape[1] / vis_frame.shape[1], key_frame.shape[0] / vis_frame.shape[0]], inv_tranformation_matrix, (court_model.shape[1], court_model.shape[0]))


        jersey_colour_prediction = np.array([[jersey_detector.detect_jersey_colour(vis_frame[y_bottom:y_top, x_bottom:x_top,]) for x_bottom, y_bottom, x_top, y_top in player_detections.pred_boxes.tensor.numpy().astype(int)]])
        
        court_model_copy = court_model.copy() 
        court_model_copy = cv2.copyMakeBorder(court_model,800,40,40,40,cv2.BORDER_CONSTANT,value=(255, 255, 255))

        for i in range(len(transformed_points[0])):
            court_model_copy = cv2.circle(court_model_copy, (int(transformed_points[0][i][0]+40), int(transformed_points[0][i][1]+800)), 4, [int(value) for value in jersey_colour_prediction[0][i]], -1)
        for i in range(len(corner_points[0])):
            colour = (255, 0, 0)
            court_model_copy = cv2.circle(court_model_copy, (int(corner_points[0][i][0]+40), int(corner_points[0][i][1]+800)), 4, colour, -1)
        vis_frame = possession_analysis.analyse_frame(ball_detections.pred_boxes.tensor.numpy().astype(int), player_detections.pred_boxes.tensor.numpy().astype(int), jersey_colour_prediction, scene_type, vis_frame)
        territory_analysis.analyse_frame(transformed_points[0], scene_type)
        log_to_file(scene_type, previous_scene, possession_analysis, territory_analysis, j)
        previous_scene = scene_type
        if data["video"]["verbose"]: 
            cv2.imshow("frame", ResizeWithAspectRatio(vis_frame, width=512))
            cv2.imshow("model", ResizeWithAspectRatio(court_model_copy, width=750))
            cv2.waitKey(1)

        j += 1
    
    video.release()
    territory_analysis._display_graphs()
    teamA_poss, total_frames = possession_analysis.possession_stats()
    territoy_stats = territory_analysis.teamA_territory(weighted=False)
    weighted_territory_stats = territory_analysis.teamA_territory(weighted=True)
    stats = {
        "possession" : {
            "teamA" : teamA_poss, 
            "totalFrames" : total_frames
        },
        "territory" : {
            "standard" : territoy_stats, 
            "weighted": weighted_territory_stats
        }
    }
    with open(data["output_file"], "wb") as file: 
        pickle.dump(stats, file)

    cv2.destroyAllWindows()
