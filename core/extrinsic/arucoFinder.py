import cv2
import cv2.aruco as aruco
import numpy as np
from .extrinsicsFinderInterface import ExtrinsicFinderInterface
from core.utils import rotationVectorToEulerAngles


class ArucoFinder(ExtrinsicFinderInterface):
    def __init__(self):
        super().__init__()
        self.save_image = True
        self.image = None
        self.aruco_image_len = 0.08132 * 1000  # 81.32 mm
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters()

    def set_param(self, mtx=None, dist=None, theory_point=None):
        self.camera_mtx = mtx
        self.camera_dist = dist
        self.theory_point = theory_point
        pass

    def get_extrinsic(self, gray_image=None):
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image,
                                                              self.aruco_dict,
                                                              parameters=self.parameters)
        trans_mat = None
        angle = None
        if ids is not None:
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 83.12, self.camera_mtx, self.camera_dist)
            for i in range(rvec.shape[0]):
                # cv2.drawFrameAxes(self.frame, self.camera_mtx, self.camera_dist, rvec[i, :, :], tvec[i, :, :], 80)
                # aruco.drawDetectedMarkers(self.frame, corners)
                r = np.zeros((3, 3), dtype=np.float64)
                cv2.Rodrigues(rvec[i][0], r)
                angle = rotationVectorToEulerAngles(rvec[0])
                trans_mat = np.concatenate((np.concatenate((r, tvec[i][0].reshape(3, 1)), axis=1), [[0, 0, 0, 1]]),
                                           axis=0)
                return trans_mat, angle
        return trans_mat, angle
        pass
