# importing the module
import cv2
import numpy as np
import pickle as pkl

class KeyframePointSelector(): 
    def __init__(self, key_frame, model): 
        self.keyframe_img = key_frame
        self.model_img = model
        self.keyframe_points = []
        self.model_points = []

    def resizeWithAspectRatio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
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

        return cv2.resize(image, dim, interpolation=inter), r

    # function to display the coordinates of
    # of the points clicked on the image
    def click_event(self, event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:


            params[2].append((x*(1/params[0]),y*(1/params[0])))
            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.keyframe_img, str(int(x*(1/params[0]))) + ',' +
                        str(int(y*(1/params[0]))), (x,y), font,
                        0.8, (255, 0, 0), 2)
            cv2.imshow(params[1], self.keyframe_img if params[1] == "image" else self.model_img)
    
    def selectKeypoints(self): 
        # reading the image

        self.keyframe_img, r = self.resizeWithAspectRatio(self.keyframe_img, width=750)
        self.model_img, s = self.resizeWithAspectRatio(self.model_img, width=750)

        # displaying the image
        cv2.imshow('image', self.keyframe_img)
        cv2.imshow('model', self.model_img)

        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image', self.click_event, param=[r, "image", self.keyframe_points])
        cv2.setMouseCallback('model', self.click_event, param=[s, "model", self.model_points])
        # wait for a key to be pressed to exit
        cv2.waitKey(0)

        # with open("test.pkl", "wb") as f: 
        #     pkl.dump(points_array, f)
        # close the window
        cv2.destroyAllWindows()
        return self.keyframe_points, self.model_points

