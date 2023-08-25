import cv2
import numpy as np
from core.calibrate.laserProjectorCali import LaserProjectorCalibration

if __name__ == '__main__':
    image = cv2.imread("../cali.jpg")
    calibrator = LaserProjectorCalibration()
    mtx = np.array([[1.79697103e+03, 0.00000000e+00, 1.16728650e+03],
                    [0.00000000e+00, 1.79632799e+03, 8.53307321e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    distort = np.array([[0.05228516, -0.23618041, 0.00102657, 0.0011332, 0.33239178]])
    calibrator.set_param(7, 7, camera_mtx=mtx, camera_dist=distort)
    calibrator.calibrate(image)
