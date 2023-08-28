import cv2
import numpy as np
from .extrinsicsFinderInterface import ExtrinsicFinderInterface
from core.utils import rotationVectorToEulerAngles


class PointFinder(ExtrinsicFinderInterface):
    def __init__(self):
        super().__init__()
        self.save_image = False
        self.image = None

    def set_param(self, mtx=None, dist=None, theory_point=None):
        self.camera_mtx = mtx
        self.camera_dist = dist
        self.theory_point = theory_point
        pass

    def get_extrinsic(self, gray_image=None):
        if gray_image is None:
            return None
        self.image = gray_image
        h, w = gray_image.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_mtx, self.camera_dist, (w, h), 0, (w, h))  #
        distort_image = cv2.undistort(gray_image, self.camera_mtx, self.camera_dist, None, new_camera_mtx)
        ret, white_image = cv2.threshold(distort_image, 75, 255, cv2.THRESH_BINARY)
        self.__output_img("white_image", white_image)
        circles = cv2.HoughCircles(white_image, cv2.HOUGH_GRADIENT_ALT, 2, 25, param1=100, param2=0.8, minRadius=3,
                                   maxRadius=20)
        self.__output_circles(circles)
        if circles is None or len(circles.reshape(-1, 3)) != len(self.theory_point.reshape(-1, 3)):
            return None, None
        circle_center = np.array(circles[0]).astype(np.float32)
        circle_center = circle_center[circle_center[:, 1].argsort()[::-1]]
        for i in range(self.theory_point.shape[0]):
            start, end = i * self.theory_point.shape[1], (i + 1) * self.theory_point.shape[1]
            sub_list = circle_center[start:end]
            circle_center[start:end] = sub_list[sub_list[:, 0].argsort()]
        theory_point_list = np.float64(self.theory_point.reshape(-1, 3))
        circle_center = np.float64(circle_center[:, 0:2])
        # print(theory_point_list)
        # print(circle_center)
        success, rvec, tvec, = cv2.solvePnP(theory_point_list, circle_center, self.camera_mtx, self.camera_dist)
        r = np.zeros((3, 3), dtype=np.float64)
        angle = rotationVectorToEulerAngles(rvec)
        cv2.Rodrigues(rvec, r)
        trans_mat = np.concatenate((np.concatenate((r, tvec.reshape(3, 1)), axis=1), [[0, 0, 0, 1]]),
                                   axis=0)
        # print(trans_mat)
        return trans_mat, angle
        pass

    def __output_circles(self, circles):
        contour_image = np.zeros(self.image.shape)
        if self.save_image:
            for (x, y, r) in circles[0]:
                x = int(x)
                y = int(y)
                cv2.circle(contour_image, (x, y), int(r), (255, 255, 255), 2)
            self.__output_img("circle_image", contour_image)

    def __output_img(self, name, image):
        if self.save_image:
            cv2.imwrite("ex_" + name + ".jpg", image)
