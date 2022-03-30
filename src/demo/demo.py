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

# Helper function to correctly resize image
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

# Detectron2 model setup 
def setup_cfg(args):
    cfgs = []
    for model in args: 
        cfg = get_cfg()
        model = args[model]
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

def log_to_file(scene, previous_scene, possession, territory, i): 
    if previous_scene != scene:
        logging.info(f"Transitioned to {scene} from {previous_scene} at {floor(i/(frames_per_second*60))}m{floor(i/frames_per_second)}s. Current Possession: {possession.possession_stats()*100}. Current Territory: {territory.teamA_territory(weighted=False)}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    logging.basicConfig(filename='test.log', format='%(levelname)s:%(message)s', level=logging.INFO)
    args = get_parser().parse_args()
    with open(args.config_file, "r") as yamlfile: 
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)

    cfg = setup_cfg(data["detectron2"])

    court_model = cv2.imread(data["field_registration"]["model"])
    key_frame = cv2.imread(data["field_registration"]["frame"])
    corner_points = np.zeros((1,4))
    previous_scene = None
    
    demo = VisualizationDemo(cfg)
    field_registor = SportsFieldRegistrator(data["field_registration"]["frame"], data["field_registration"]["model"], data["field_registration"]["simulation_ranges"], data["field_registration"]["line_mask"], data["video"]["verbose"])
    jersey_detector = JerseyColourDetection(
        data["jersey_detection"]["teamA_colour"], 
        data["jersey_detection"]["teamB_colour"], 
        data["jersey_detection"]["offical_colour"], 
    )
    possession_analysis = PossessionAnalysis(data["jersey_detection"]["teamA_colour"][2])
    territory_analysis = TerritoryAnalysis()
    scene_classification = SceneClassification([(33, 0, 0), (70, 255, 255)], 0.5, 0.02)
    
    field_registor.generateSimulatedEdgeMaps(data["field_registration"]["keypoints"])

    assert os.path.isfile(data["video"]["input"])
    video = cv2.VideoCapture(data["video"]["input"])
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    basename = os.path.basename(data["video"]["input"])
    codec, file_ext = (
        ("mp4v", ".mp4")
    )
    
    j = 0

    # The culmination of many a mental breakdown 
    for vis_frame, predictions in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
        player_detections = predictions[0]
        ball_detections = predictions[1]
        # Classify scene and skip if it doesn't contain the playing field
        scene_type = scene_classification.frame_classification(vis_frame, [vis_frame[y_bottom:y_top, x_bottom:x_top] for x_bottom, y_bottom, x_top, y_top in player_detections.pred_boxes.tensor.numpy().astype(int)])
        if scene_type == SceneClass.NON_FIELD: 
            continue
        
        # Get pitch registration and inverse homography
        tranformation_matrix, _ = field_registor.pitch_registration(vis_frame, np.array([[x_bottom, y_bottom, x_top, y_top] for x_bottom, y_bottom, x_top, y_top in player_detections.pred_boxes.tensor.numpy()]), scene_type, corner_points)
        inv_tranformation_matrix = np.linalg.inv(tranformation_matrix)

        # Get centre bottom point of players bounding boxes and transform by returned homography
        center_bottom_prediction = np.array([[[x_top + (x_bottom - x_top)/2, y_bottom] for x_top, y_top, x_bottom, y_bottom in player_detections.pred_boxes.tensor.numpy()]])
        if center_bottom_prediction.size == 0: 
            transformed_points = np.array([[]])
        else: 
            transformed_points = cv2.perspectiveTransform(center_bottom_prediction * [key_frame.shape[1] / vis_frame.shape[1], key_frame.shape[0] / vis_frame.shape[0]], inv_tranformation_matrix, (court_model.shape[1], court_model.shape[0]))

        # Get team classification based on jersey colour
        jersey_colour_prediction = np.array([[jersey_detector.detect_jersey_colour(vis_frame[y_bottom:y_top, x_bottom:x_top,]) for x_bottom, y_bottom, x_top, y_top in player_detections.pred_boxes.tensor.numpy().astype(int)]])
        
        #
        vis_frame = possession_analysis.analyse_frame(ball_detections.pred_boxes.tensor.numpy().astype(int), player_detections.pred_boxes.tensor.numpy().astype(int), jersey_colour_prediction, scene_type, vis_frame)
        territory_analysis.analyse_frame(transformed_points[0], scene_type)
        
        # Transform corners of the current frame by homography returned by sport field registration
        vis_frame_corners = np.array([[(0, vis_frame.shape[0]), (0, 0), (vis_frame.shape[1], 0), (vis_frame.shape[1], vis_frame.shape[0])]]) * [key_frame.shape[1] / vis_frame.shape[1], key_frame.shape[0] / vis_frame.shape[0]]
        corner_points = cv2.perspectiveTransform(vis_frame_corners.astype(float), inv_tranformation_matrix, (court_model.shape[1], court_model.shape[0]))
        
        log_to_file(scene_type, previous_scene, possession_analysis, territory_analysis, j)
        previous_scene = scene_type
        j += 1

        # Display pitch registration and current frame
        if data["video"]["verbose"]:
            court_model_copy = court_model.copy() 
            court_model_copy = cv2.copyMakeBorder(court_model,800,40,40,40,cv2.BORDER_CONSTANT,value=(255, 255, 255))
            for i in range(len(transformed_points[0])):
                court_model_copy = cv2.circle(court_model_copy, (int(transformed_points[0][i][0]+40), int(transformed_points[0][i][1]+800)), 4, [int(value) for value in jersey_colour_prediction[0][i]], -1)
            for i in range(len(corner_points[0])):
                colour = (255, 0, 0)
            court_model_copy = cv2.circle(court_model_copy, (int(corner_points[0][i][0]+40), int(corner_points[0][i][1]+800)), 4, colour, -1) 
            cv2.imshow("frame", ResizeWithAspectRatio(vis_frame, width=512))
            cv2.imshow("model", ResizeWithAspectRatio(court_model_copy, width=750))
            cv2.waitKey(1)
    
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
