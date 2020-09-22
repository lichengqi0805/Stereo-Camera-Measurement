import numpy as np
import cv2

#双目相机参数
class stereoCameral(object):
    def __init__(self):

        #左相机内参数
        self.cam_matrix_left = np.array([[869.01155223, 0. ,655.40791121], [0. ,863.80440148 ,412.05627691], [0., 0., 1.]])
        #右相机内参数
        self.cam_matrix_right = np.array([[849.06156676, 0. ,679.40355396], [0., 844.69413042, 405.9092638], [0., 0., 1.]])

        #左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[-8.04406637e-01 ,4.13138039e+00 ,-2.34503273e-02,  4.65657331e-03]])
        self.distortion_r = np.array([[-0.4511927, -0.28352434, -0.01746639, -0.00436867,  1.05200469]])

        #旋转矩阵
        om = np.array([9.99652544e-01, 9.99899066e-01, 9.99551990e-01])
        self.R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
        #平移矩阵
        self.T = np.array([-0.64951112, 0.0093585, -0.106197])
