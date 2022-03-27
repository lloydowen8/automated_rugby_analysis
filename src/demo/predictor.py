# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import numpy as np
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


class VisualizationDemo(object):
    def __init__(self, cfgs, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = []
        self.predictor = []
        for i in range(0, len(cfgs)):
            metadata = MetadataCatalog.get(
                cfgs[i].DATASETS.TEST[0] if len(cfgs[i].DATASETS.TEST) else "__unused"
            )
            self.metadata.append(metadata)
            
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.frame = 0

        for i in range(0, len(cfgs)):
            self.predictor.append(DefaultPredictor(cfgs[i]))
        
        self.video_visualizer = []
        for i in range(0, len(self.metadata)):
            self.video_visualizer.append(VideoVisualizer(self.metadata[i], self.instance_mode))

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """


        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            instance_predictions = []
            for i in range(0, len(predictions)): 
                if "instances" in predictions[i]:
                    pred = predictions[i]["instances"].to(self.cpu_device)
                    instance_predictions.append(pred)

                    vis_frame = self.video_visualizer[i].draw_instance_predictions(frame, pred)
                else:
                    continue
                # Converts Matplotlib RGB format to OpenCV BGR format
                # frame = vis_frame.get_image()
        
            self.frame += 1
            return cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR), instance_predictions

        frame_gen = self._frame_from_video(video)
        
        for frame in frame_gen:
            yield process_predictions(frame, [self.predictor[i](frame) for i in range(0, len(self.predictor))])


    def run_on_image(self, image):
        vis_output = None
        predictions = self.predictor[0](image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)

        return vis_output, instances
