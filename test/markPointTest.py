import cv2
import numpy as np
from core.extrinsic.pointFinder import PointFinder

if __name__ == '__main__':
    image = cv2.imread("../2.jpg")
    finder = PointFinder()
    mtx = np.array([[1.79697103e+03, 0.00000000e+00, 1.16728650e+03],
                    [0.00000000e+00, 1.79632799e+03, 8.53307321e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    distort = np.array([[0.05228516, -0.23618041, 0.00102657, 0.0011332, 0.33239178]])
    theory_point = np.array([[[0, 0, 0], [184.4, 0, 0]], [[0, 265, 0], [184.4, 265, 0]]])
    finder.set_param(mtx, distort, theory_point)
    finder.save_image = True
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    extrinsic = finder.get_extrinsic(gray_image)
    print(extrinsic[0])
