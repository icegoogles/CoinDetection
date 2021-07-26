import cv2
import cv2 as cv
import numpy as np

import findlines as fl
import perspective
from distortion_calibration import undistort

import config as cfg

class calibration():

    scale = None  # [pixel/mm]
    cornerPoints = None  # cornerpoints of square

    # for anti fisheye warp
    cam_matrix = None
    dist_param = None

    # Methods
    def __init__(self) -> None:
        pass

    def preprocess(self, img):
        """Transforms the image according to the given options"""

        #cv2.imshow("before preprocess", img)

        if self.cam_matrix is not None and self.dist_param is not None:
            # remove fisheye effect from image
            img = undistort(self.cam_matrix, self.dist_param, img)
            

        if self.cornerPoints is not None:  # remove perspective warp
            img = perspective.warp_image(img, self.cornerPoints, cfg.warpscale)

        #cv2.imshow("after preprocess", img)
        
        return img


    def calibration(self, img, sideLengthSquareMM, calibrationType:str='area'):
        """calibrate setup with a given calibrationtype.
        Check the function for all calibration types."""

        if calibrationType == 'area':
            validateImg = self._scaling_trough_area(img, sideLengthSquareMM, imgIsBW=False)
            validateImg = cv2.cvtColor(validateImg, cv2.COLOR_GRAY2BGR)
            return validateImg

        elif calibrationType == 'warp':
            self.cornerPoints = fl.find_corners(img)
            validateImg = fl.draw_corners(img, self.cornerPoints)

            warpedImg = perspective.warp_image(img, self.cornerPoints, cfg.warpscale)

            self._scaling_trough_area(warpedImg, sideLengthSquareMM, imgIsBW=False)

            return validateImg
            
        else:
            assert False, "WARNING! No calibration done, unknown calibration type!"
        


    def _scaling_trough_area(self, img, sideLengthSquareMM, imgIsBW=False):
        """
        :img: image from the webcam
        :sideLengthSquareMM: sidelength of (dark) square in the image
        :imgIsBW: = False   if set to True the binarysation of the Image is not done by this function
        """

        if imgIsBW == False:
            # OTSU method for binarysation
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kernel_size = 5
            blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
            ret, binImg = cv2.threshold(blur_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            binImg = img

        # Convert binary image to ones and zeros
        img01 = binImg / np.amax(binImg)
        img01 = img01.astype('uint8')

        A = np.sum(img01) # area in Px
        a = A**0.5  # sidelength in Px

        self.scale = a / sideLengthSquareMM

        return binImg
   


if __name__ == "__main__":
    path = "ideal3.jpg"

    img = cv.imread(path)

    a = calibration()

    a.scaling_trough_area(img, 80)
    print("")