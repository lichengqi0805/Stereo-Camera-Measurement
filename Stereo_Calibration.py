from stereovision.calibration import *
import numpy as np
import os
import cv2

current_dir = os.getcwd()
num = 113
data_dir = os.path.join(current_dir, 'Dataset',str(num))
file_num = 2
print("Data from directory: ", data_dir)


calibrator = StereoCalibrator(7, 9, 2.15, [1280, 960])
img_left_dir = os.path.join(data_dir, 'cam_1_'+str(file_num)+'.jpeg')
img_left = cv2.imread(img_left_dir)
img_right_dir = os.path.join(data_dir, 'cam_2_'+str(file_num)+'.jpeg')
img_right = cv2.imread(img_right_dir)

calibrator.add_corners((img_left, img_right))

calibration = StereoCalibration()
calib_loaded = StereoCalibration(input_folder=data_dir)
