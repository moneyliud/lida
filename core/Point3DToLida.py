import math

from .lidaGenerator import LidaFile
from .imageConverter import ImageILDAConverter
from .utils import point_to_angle
import numpy as np
from numpy import array


class Point3DToLida(ImageILDAConverter):
    def __init__(self):
        super().__init__()
        self.camera_project_trans_mtx: array = np.diag([1, 1, 1])
        self.camera_trans_mtx: array = np.diag([1, 1, 1])
        self.x_angle_interval = [-16.7, 16.7]
        self.y_angle_interval = [-16.7, 16.7]
        self.point_max = 65535
        self.lida_file = LidaFile()
        self.point_size = 4
        self.scales = 1.0
        self.contour_list = []

    def add_point(self, point, color=None):
        if color is None:
            color = [255, 255, 255]
        contour = []
        x, y = self.point3d_to_image_point(point)
        # 上半圆
        for i in range(-int(self.point_size / 2), int(self.point_size / 2) + 1):
            for j in range(int(self.point_size / 2) + 1):
                x1 = x + i
                y1 = y + j
                radius = math.pow((x1 - x) ** 2 + (y1 - y) ** 2, 0.5)
                if self.point_size / 2 - 1 < radius <= self.point_size / 2:
                    contour.append([x1, y1, color])
                    break
        # 下半圆
        for i in range(int(self.point_size / 2), -int(self.point_size / 2) - 1, -1):
            for j in range(0, -int(self.point_size / 2) - 1, -1):
                x1 = x + i
                y1 = y + j
                if self.point_size / 2 - 1 < math.pow((x1 - x) ** 2 + (y1 - y) ** 2, 0.5) <= self.point_size / 2:
                    contour.append([x1, y1, color])
                    break
        self.contour_list.append(contour)
    pass

    def add_contour(self, contour, color):
        contour_format = []
        if contour is not None and len(contour) > 0:
            for point in contour:
                format_point = self.point3d_to_image_point(point).append(color)
                contour_format.append(format_point)
            self.contour_list.append(contour_format)
        pass

    def point3d_to_image_point(self, point):
        x, y = 0, 0
        if self.camera_trans_mtx is not None:
            actual_point = np.dot(self.camera_trans_mtx, point)
            actual_point = np.dot(self.camera_project_trans_mtx, actual_point)
            alpha, beta = point_to_angle(actual_point[0:3])
            if alpha is not None:
                x = int(alpha / self.x_angle_interval[1] * int(self.point_max / 2)) + int(self.point_max / 2)
            if beta is not None:
                # 投影仪二维右手系转图像二维左手系
                y = int(beta / self.y_angle_interval[1] * int(self.point_max / 2)) + int(self.point_max / 2)
        return [x, y]

    def to_bytes(self):
        self.contour_list = np.array(self.contour_list, dtype=object)
        points = self.convert_contour_to_projection()
        self.lida_file.name = "actual"
        self.lida_file.company = "buz141"
        self.lida_file.new_frame()
        print(len(points))
        for i in range(len(points)):
            point = points[i]
            self.lida_file.add_point(point[0], point[1], point[2], point[3], point[4])
        return self.lida_file.to_bytes()
