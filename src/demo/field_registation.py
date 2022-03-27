import cv2
from matplotlib import pyplot as plt
import numpy as np
import pickle
import math
from keyframe_point_selector import KeyframePointSelector
from scipy.spatial import distance
from scene_classification import SceneClass

class SportsFieldRegistrator(object): 
    def __init__(self, key_frame_file, model_file, edge_map_simulation_ranges, line_mask, verbose=False):
        self.hog = cv2.HOGDescriptor()
        self.court_model = cv2.imread(model_file)
        self.key_frame = cv2.imread(key_frame_file)
        self.line_mask = (tuple(line_mask[0]), tuple(line_mask[1]))
        self.previous_transformation = np.zeros((3,3))
        self.previous_scene_class = SceneClass.NON_FIELD
        self.verbose = verbose
        self.inner_edge_map = None
        self.use_inner_edge_map = False
        self.edge_map_simulation_ranges = edge_map_simulation_ranges

    def _getKeyframePoints(self, keyframe, model): 
        point_selector = KeyframePointSelector(keyframe, model)
        return point_selector.selectKeypoints()

    def _keyframe_homography(self, key_frame_points, model_points): 
        model_points = np.array(model_points)
        key_frame_points = np.array(key_frame_points)
        M, _ = cv2.findHomography(model_points, key_frame_points, cv2.RANSAC, 5.0)

        x_offset = 0
        y_offset = -0
        shift=np.array([[1, 0, x_offset], [0, 1, y_offset], [0 , 0 , 1]])
        
        return M

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

    def _simulateCameraPan(self, corners, origin, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        for i, point in enumerate(corners[0]):
            ox, oy = origin
            px, py = point

            qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
            qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
            corners[0][i] = qx, qy
        return corners
    
    def _simulateCameraTilt(self, corners, length):
        left_line = corners[0][:2] 
        right_line = corners[0][:1:-1]
        tilted_lines = []
        for line in [left_line, right_line]:
            #print(line) 
            pt1 = line[0]
            pt2 = line[1]
            px, py = pt1
            qx, qy = pt2
            v = [px - qx, py - qy]

            lenAB = math.sqrt(pow(px - qx, 2) + pow(py - qy, 2))
            norm_v = [_v*length/lenAB for _v in v]

            pt1 = np.array(pt1 + norm_v)
            pt2 = np.array(pt2 + norm_v)
            tilted_lines.append([pt1, pt2])
        
        tilted_lines = np.array(tilted_lines).reshape(1,4,2)
        tilted_lines[0][2:] = tilted_lines[0][:1:-1]
        return tilted_lines
        
    def _simulateCameraZoom(self, camera_pos, corners, scale_factor, offset = 0): 
        centroid = corners[0].mean(axis=0)

        v = [centroid[0] - camera_pos[0], centroid[1] - camera_pos[1]]
        lenAB = math.sqrt(pow(centroid[0] - camera_pos[0], 2) + pow(centroid[1] - camera_pos[1], 2))
        norm_v = [_v*offset/lenAB for _v in v]
        centroid = centroid + norm_v

        for i, point in enumerate(corners[0]): 
            corners[0][i][0] = scale_factor * (point[0] - centroid[0]) + centroid[0]
            corners[0][i][1] = scale_factor * (point[1] - centroid[1]) + centroid[1]
        return corners
    
    def _getCameraPos(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return np.array((x, y))

    def _simulateCamera(self, camera_corners, homography, simlulation_ranges, zoom_offset = 0): 
        camera_pos = self._getCameraPos(camera_corners[0][:2], camera_corners[0][2:])
        edge_map_hog_dict = []
        pan_range = simlulation_ranges["pan_range"]
        tilt_range = simlulation_ranges["tilt_range"]
        zoom_range = simlulation_ranges["zoom_range"]
        for i in np.arange(pan_range[0], pan_range[1], pan_range[2]):
            for j in np.arange(tilt_range[0], tilt_range[1], tilt_range[2]):
                for k in np.arange(zoom_range[0], zoom_range[1], zoom_range[2]):
                    # print(f"\r{iter}", end="", flush=True)
                    new_corners = self._simulateCameraPan(camera_corners.copy(), camera_pos, math.radians(i/2))
                    new_corners = self._simulateCameraTilt(new_corners, j)
                    new_corners = self._simulateCameraZoom(camera_pos, new_corners, 1/k, zoom_offset)
                    N, _ = cv2.findHomography(camera_corners, new_corners, cv2.RANSAC, 5.0)
                    dst2 = cv2.cvtColor(self.court_model, cv2.COLOR_RGB2GRAY)
                    dst2 = cv2.Canny(dst2, 10, 150, apertureSize=3)
                    dst2 = cv2.warpPerspective(dst2, np.matmul(homography, np.linalg.inv(N)), (self.court_model.shape[1], self.court_model.shape[0]))
                    if self.verbose: 
                        cv2.imshow("Dictionary generation", self._resizeWithAspectRatio(dst2, width=512))
                        cv2.waitKey(1)
                    dst2 = self._resizeWithAspectRatio(dst2, width=256)

                    h_output = self.hog.compute(dst2)
                    edge_map_hog_dict.append([h_output, np.matmul(homography, np.linalg.inv(N))])
        return edge_map_hog_dict

    def generateSimulatedEdgeMaps(self, keypoints = None): 
        if keypoints == None: 
            key_frame_points_middle, model_points = self._getKeyframePoints(self.key_frame, self.court_model)
        else: 
            with open(keypoints["model"], "rb") as file:
                model_points = pickle.load(file)
            with open(keypoints["frame"], "rb") as file:
                key_frame_points_middle = pickle.load(file)

        input_shape = self.key_frame.shape

        court_model = self.court_model
        court_model_grey = cv2.cvtColor(court_model, cv2.COLOR_RGB2GRAY)
        court_model_canny = cv2.Canny(court_model_grey, 10, 150, apertureSize=3)

        M = self._keyframe_homography(key_frame_points_middle, model_points)
        dst = cv2.warpPerspective(court_model_canny, M, (input_shape[1], input_shape[0]))

        corner_points = np.array([[(0, dst.shape[0]), (0, 0), (dst.shape[1], 0), (dst.shape[1], dst.shape[0])]])

        corners = cv2.perspectiveTransform(corner_points.astype(float), np.linalg.inv(M))



        self.edge_map_hog_dict = self._simulateCamera(corners, M, self.edge_map_simulation_ranges)

    def loadEdgeMaps(edge_map_file): 
        with open("edge_map_hog_dict.pkl", "rb") as f: 
            edge_map_dict = pickle.load(f)
        return edge_map_dict


    def _preprocess_frame(self, frame, masks): 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, self.line_mask[0], self.line_mask[1]) 
        frame = cv2.bitwise_and(frame, frame, mask=mask_green)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame,(7,7),cv2.BORDER_DEFAULT)
        frame = cv2.Canny(frame, 0, 100, apertureSize=5)
        
        for mask in masks: 
            frame[int(mask[1]):int(mask[3]),int(mask[0]):int(mask[2])] = np.zeros((int(mask[3])-int(mask[1]), int(mask[2]) - int(mask[0])))
        
        frame = cv2.resize(frame, (self.court_model.shape[1], self.court_model.shape[0]))
        frame = self._resizeWithAspectRatio(frame, width=256)
        return frame

    def _mrf_optimisation(self, neighbour_dists, transformations, previous_transformation): 
        if not previous_transformation.any(): 
            return np.argmin(self._data_function(neighbour_dists))
        else:
            return np.argmin(self._data_function(neighbour_dists) + self._smoothness_function(previous_transformation/np.linalg.norm(previous_transformation), [i/j for i,j in zip(transformations, [np.linalg.norm(transformation) for transformation in transformations])]))

    def _data_function(self,neighbour_dists):
        return np.log(neighbour_dists)

    def _smoothness_function(self, previous_transformation, transformations):
        return np.array([distance.euclidean(previous_transformation.flatten(), transformation.flatten()) for transformation in transformations])

    def find_best_edge_map(self, frame, masks, scene_class, corner_points):
        frame = self._preprocess_frame(frame, masks)
        h_input = self.hog.compute(frame)

        zoomin_transition = self.previous_scene_class == SceneClass.WIDE_VIEW and scene_class == SceneClass.CLOSE_VIEW
        zoomout_transition = self.previous_scene_class == SceneClass.CLOSE_VIEW and scene_class == SceneClass.WIDE_VIEW
        edge_map_simulation_ranges = {
            "pan_range" : [-6, 6, 1.5], 
            "tilt_range": [-100, 200, 50], 
            "zoom_range": [3, 5, 1]
        }
        
        if zoomin_transition: 
            self.use_inner_edge_map = True
            self.inner_edge_map = self._simulateCamera(corner_points, self.previous_transformation, edge_map_simulation_ranges, 0)
        elif zoomout_transition: 
            self.use_inner_edge_map = False
            self.inner_edge_map = None

        edge_map_hog_dict = self.edge_map_hog_dict if not self.use_inner_edge_map else self.inner_edge_map

        min_dist_array = []
        closest_homographies = []
        for h_output, transform_matrix in edge_map_hog_dict:
            dist = distance.euclidean(h_input, h_output)
            if len(min_dist_array) < 4: 
                min_dist_array.append(dist)
                closest_homographies.append(transform_matrix)
                if len(min_dist_array) == 4: 
                    min_dist_array.sort()
                    max_pointer = len(min_dist_array)-1
            if len(min_dist_array) >= 4: 
                if dist < min_dist_array[max_pointer]:
                    min_dist_array[max_pointer] = dist
                    closest_homographies[max_pointer] = transform_matrix
                    max_pointer = (max_pointer + 1) % len(min_dist_array)

        self.previous_scene_class = scene_class
        tranfromation_idx = self._mrf_optimisation(min_dist_array, closest_homographies, self.previous_transformation)
        return closest_homographies[tranfromation_idx], frame

    def pitch_registration(self, frame, masks, scene_class, corner_points): 
        transform_matrix, frame = self.find_best_edge_map(frame, masks, scene_class, corner_points)
        self.previous_transformation = transform_matrix
        return transform_matrix, frame 