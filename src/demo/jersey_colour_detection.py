import cv2
from PIL import Image
from matplotlib import cm
import numpy as np

class JerseyColourDetection(): 
    def __init__(self, teamA, teamB, officals): 
        self.team_one_colour = np.array(teamA)
        self.team_two_colour = np.array(teamB)
        self.offical_colour = np.array(officals)
        self.team_colours = [self.team_one_colour, self.team_two_colour, self.offical_colour]

    def _resizeWithAspectRatio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
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

    def detect_jersey_colour(self, cropped_frame): 
        max_colour_ratio = 0
        team_colour = [255, 255, 255]
        hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
        for colour_range in self.team_colours: 
            masked_image = cv2.inRange(hsv, colour_range[0], colour_range[1])
            colour_ratio = cv2.countNonZero(masked_image) / masked_image.size
            if colour_ratio > max_colour_ratio: 
                max_colour_ratio = colour_ratio
                team_colour = colour_range[2]
        return team_colour 