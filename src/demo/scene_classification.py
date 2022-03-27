import cv2
import numpy as np
from enum import Enum

class SceneClass(Enum): 
    NON_FIELD = "Non-field"
    WIDE_VIEW = "Wide-view"
    CLOSE_VIEW = "Close-view"


class SceneClassification(): 

    def __init__(self, green_mask_range, green_pixel_threshold, bounding_box_threshold): 
        self.green_mask_range = green_mask_range
        self.green_pixel_threshold = green_pixel_threshold
        self.bounding_box_threshold = bounding_box_threshold

    def _green_pixel_ratio(self, frame): 
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masked_frame = cv2.inRange(hsv_frame, self.green_mask_range[0], self.green_mask_range[1])
        green_pixel_ratio = np.count_nonzero(masked_frame) / masked_frame.size
        return green_pixel_ratio

    def _avg_bounding_box_ratio(self, frame, bounding_boxes): 
        bounding_box_ratio = 0
        for bounding_box in bounding_boxes: 
            if bounding_box.size / frame.size > bounding_box_ratio:
                bounding_box_ratio = bounding_box.size / frame.size
            
        return bounding_box_ratio

    def frame_classification(self, frame, bounding_boxes): 
        green_pixel_ratio = self._green_pixel_ratio(frame)
        bounding_box_ratio = self._avg_bounding_box_ratio(frame, bounding_boxes)

        if green_pixel_ratio < self.green_pixel_threshold: 
            return SceneClass.NON_FIELD
        elif bounding_box_ratio > self.bounding_box_threshold: 
            return SceneClass.CLOSE_VIEW
        else:
            return SceneClass.WIDE_VIEW
    