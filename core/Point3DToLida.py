import math

from .lidaGenerator import LidaFile
from .imageConverter import ImageILDAConverter
from .utils import point_to_angle
import numpy as np
from numpy import array


class PointAddInterface:
    def __init__(self):
        pass

    def get_contour(self, x, y, color):
        pass


class Point3DToLida(ImageILDAConverter):
    def __init__(self):
        super().__init__()
        self.camera_project_trans_mtx: array = np.diag([1, 1, 1])
        self.camera_trans_mtx: array = np.diag([1, 1, 1])
        self.projector_in_mtx: array = np.diag([1, 1, 1])
        self.x_angle_interval = [-16.7, 16.7]
        self.y_angle_interval = [-16.7, 16.7]
        self.point_max = 65535
        self.lida_file = LidaFile()
        self.convert_type = "camera"
        self.scales = 1.0
        self.contour_list = []
        self.point_add_strategy: PointAddInterface = PointAddInterface()

    def set_strategy(self, strategy: PointAddInterface):
        self.point_add_strategy = strategy

    def add_point(self, point, color=None):
        if color is None:
            color = [255, 255, 255]
        x, y = self.point3d_to_image_point(point)
        contour = self.point_add_strategy.get_contour(x, y, color)
        self.contour_list.append(contour)
        pass

    def add_contour(self, contour, color):
        contour_format = []
        if contour is not None and len(contour) > 0:
            for point in contour:
                format_point = self.point3d_to_image_point(point)
                format_point.append(color)
                contour_format.append(format_point)
            self.contour_list.append(contour_format)
        pass

    def point3d_to_image_point(self, point):
        if self.convert_type == "camera":
            return self.point3d_to_image_point_camera(point)
        if self.convert_type == "origin":
            return self.point3d_to_image_point_origin(point)

    def point3d_to_image_point_origin(self, point):
        x, y = 0, 0
        if self.camera_trans_mtx is not None:
            actual_point = np.dot(self.camera_trans_mtx, point)
            actual_point = np.dot(self.camera_project_trans_mtx, actual_point)
            x, y = actual_point[0], actual_point[1]
            alpha, beta = point_to_angle(actual_point[0:3])
            if alpha is not None:
                x = int(alpha / self.x_angle_interval[1] * int(self.point_max / 2)) + int(self.point_max / 2)
            if beta is not None:
                # 投影仪二维右手系转图像二维左手系
                y = int(beta / self.y_angle_interval[1] * int(self.point_max / 2)) + int(self.point_max / 2)
            print("x,y=", x, y)
        return [x, y]

    def point3d_to_image_point_camera(self, point):
        x, y = 0, 0
        if self.camera_trans_mtx is not None:
            actual_point = np.dot(self.camera_trans_mtx, point)
            actual_point = np.dot(self.camera_project_trans_mtx, actual_point)
            actual_point = np.dot(self.projector_in_mtx, actual_point[0:3])
            actual_point = actual_point / actual_point[2]
            # print(actual_point)
            x, y = actual_point[0], actual_point[1]
            # alpha, beta = point_to_angle(actual_point[0:3])
            # if alpha is not None:
            #     x = int(alpha / self.x_angle_interval[1] * int(self.point_max / 2)) + int(self.point_max / 2)
            # if beta is not None:
            #     # 投影仪二维右手系转图像二维左手系
            #     y = int(beta / self.y_angle_interval[1] * int(self.point_max / 2)) + int(self.point_max / 2)
            print("x,y=", x, y)
        return [x, y]

    def new_frame(self, duration=0):
        self.lida_file.name = "actual"
        self.lida_file.company = "buz141"
        self.lida_file.new_frame()
        self.lida_file.frames[self.lida_file.cur_frame_index].header.duration = duration

    def end_frame(self):
        self.contour_list = np.array(self.contour_list, dtype=object)
        points = self.convert_contour_to_projection()
        for i in range(len(points)):
            point = points[i]
            self.lida_file.add_point(point[0], point[1], point[2], point[3], point[4])
        print("frame point length= ", len(self.lida_file.frames[self.lida_file.cur_frame_index].points))
        self.contour_list = []
        self.points = []
        self.org_points = []
        self.image = None
        self.scales = 1.0

    def add_image(self, image):
        points = self.convert(image)
        for i in range(len(points)):
            point = points[i]
            self.lida_file.add_point(point[0], point[1], point[2], point[3], point[4])
        #
        # self.contour_list = []
        # self.points = []

    def to_bytes(self):
        return self.lida_file.to_bytes()
