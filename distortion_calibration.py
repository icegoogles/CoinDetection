import numpy as np
import cv2 as cv
import glob
import os
from tkinter import messagebox
import config as cfg
import classes


# Delete previous calibration images
def delete_calib_imgs():
    delete_all = glob.glob('*.jpg')
    for i in delete_all:
        try:
            os.remove(str(i))
        except: pass

    


def start_dist_calib(images, calibrationClass):
    #################### FIND CHESSBOARD CORNERS - objPoints AND imgPoints ####################

    #chessboard_size = (17, 9) #(Spalten(Quadrate - horizontal-1), Zeilen (Quadrate - vertikal-1))
    #frame_size = (1920, 1080) #Kamera-Auflösung

    # Termination Criterai for Sub-Pixels
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare Object Points (Coordinates: (0,0,0), (1,0,0), (2,0,0) ...)
    objp = np.zeros((cfg.chessboard_size[0] * cfg.chessboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:cfg.chessboard_size[0], 0:cfg.chessboard_size[1]].T.reshape(-1,2)

    # Array to store object Points and image Points from all the Images
    obj_points = [] # 3D Points in real world space
    img_points = [] # 2D Points in image plane


    for image in images:
        print(image)
        img = cv.imread(image)
        #cv.imshow('img', img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find Chessboard Corners
        ret, corners = cv.findChessboardCorners(gray, cfg.chessboard_size, None)

        # If found, add object points & image points (after refining them)
        if ret == True:

            obj_points.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            img_points.append(corners)

            #Draw and display the corners
            cv.drawChessboardCorners(img, cfg.chessboard_size, corners2, ret)
            img = cv.resize(img, (480, 360), interpolation=cv.INTER_AREA)
            cv.imshow('Chessboard detected!', img)
            cv.waitKey(100)
        else:
            print("No Chessboard found!")

    cv.destroyAllWindows()


    #################### Calibration ####################
    try:
        ret, camera_matrix, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, cfg.frame_size, None, None)
        print("\nKamera Matrix:\n", camera_matrix)
        print("\nVerzerrungsparameter:\n", dist)
        print("\nRotationsvektoren:\n", rvecs)
        print("\nTranslationsvektoren:\n", tvecs)

        # Previous fisheye calibration storage
        # np.savetxt("camMatrix.csv", camera_matrix, delimiter=',')
        # np.savetxt("distParam.csv", dist, delimiter=',')

        # write to calibration class
        calibrationClass.cam_matrix = camera_matrix.tolist()
        calibrationClass.dist_param = dist[0].tolist()

        done = True
    except:
        
        done = False
    return done


def undistort(camera_matrix, dist_param, img):
    ################### Entzerrung ####################
    #camera_matrix = np.loadtxt("camMatrix.csv", delimiter=',')
    #dist_param = np.loadtxt("distParam.csv", delimiter=',')
    h, w = img.shape[:2]

    camera_matrix = np.array(camera_matrix, dtype='float32')
    dist_param = np.array(dist_param, dtype='float32')

    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_param, (w,h), 1, (w,h))

    # Entzerren
    dst = cv.undistort(img, camera_matrix, dist_param, None, new_camera_matrix)
    #Crop image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

    
# rvecs_shape = np.array(rvecs).shape
# save_rvecs = np.reshape(rvecs, (rvecs_shape[0], rvecs_shape[1]))
# np.savetxt("rotVect.csv", save_rvecs, delimiter=',')

# tvecs_shape = np.array(tvecs).shape
# save_tvecs = np.reshape(tvecs, (tvecs_shape[0], tvecs_shape[1]))
# np.savetxt("transVect.csv", save_tvecs, delimiter=',')

#myCalibration.store_as_list(camera_matrix, dist, rvecs, tvecs)


#################### Entzerrung ####################

# img = cv.imread('1.jpg') #Irgendein Kalibrierungsbild
# h, w = img.shape[:2]
# new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist, (w,h), 1, (w,h))

# #Entzerren
# dst = cv.undistort(img, camera_matrix, dist, None, new_camera_matrix)
# # #Crop image
# # x, y, w, h = roi
# # dst = dst[y:y+h, x:x+w]
# cv.imwrite('caliResult1.jpg', dst)

# #Reprojection Error
# mean_error = 0

# for i in range(len(obj_points)):
#     img_points2, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist)
#     error = cv.norm(img_points[i], img_points2, cv.NORM_L2)/len(img_points2)
#     mean_error += error

# print("\nAbweichung: {}".format(mean_error/len(obj_points)))
# print("\n\n\n")
