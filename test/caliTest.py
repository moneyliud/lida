import cv2
import numpy as np
import os
from core.calibrate.laserProjectorCali import LaserProjectorCalibration

if __name__ == '__main__':
    base_dir = "../caliImage/"
    paths = os.listdir(base_dir)
    images = []
    image_indexes = []
    xy_num = 3
    for path in paths:
        if not os.path.isdir(path):
            image = cv2.imread(base_dir + "/" + path)
            post_fix = path.split(".")[0][-3:]
            images.append(image)
            idx = int(post_fix) - 1
            idx_x = idx % 3
            idx_y = int(idx / 3)
            image_indexes.append([idx_x, idx_y])
    calibrator = LaserProjectorCalibration()
    mtx = np.array([[1.79697103e+03, 0.00000000e+00, 1.16728650e+03],
                    [0.00000000e+00, 1.79632799e+03, 8.53307321e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    distort = np.array([[0.05228516, -0.23618041, 0.00102657, 0.0011332, 0.33239178]])
    calibrator.init(7, 7, camera_mtx=mtx, camera_dist=distort, offset_y=17000, cali_img_interval=10000, xy_num=xy_num)
    calibrator.calibrate(images, image_indexes)
