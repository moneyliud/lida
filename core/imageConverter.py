from skimage import morphology
import numpy as np
import os
import cv2
import math
from .utils import rgb_to_ilda_color, rgb_to_7_color, is_out_of_signed_short_range
from .LIDA import COLOR


class ImageILDAConverter:
    def __init__(self):
        self.scales = 0.0
        self.laser_point_interval = 1200
        self.blank_point_interval = 500
        self.save_image = False
        self.skeleton = None
        self.image = None
        self.points = []
        self.org_points = []
        self.contour_use_flag = None
        self.contour_list = None
        self.end_blank_repeat_num = 3
        self.tiny_contour_repeat_num = 4
        self.dis_threshold = 2000
        pass

    def convert(self, image):
        self.image = image
        self.scales = 65536.0 / image.shape[0]
        self.__find_skeleton()
        # hierarchy [后一个轮廓，前一个轮廓，父轮廓，内嵌轮廓]
        self.contour_list, hierarchy = cv2.findContours(self.skeleton, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        return self.convert_contour_to_projection()

    def convert_contour_to_projection(self):
        if len(self.contour_list) == 0:
            return []
        else:
            contour_list_tmp = []
            for i in range(len(self.contour_list)):
                # if (hierarchy[0][i][3] == -1 and hierarchy[0][i][2] == -1) or hierarchy[0][i][3] >= 0:
                if len(self.contour_list[i].shape) > 2:
                    contour_list_tmp.append(np.squeeze(self.contour_list[i], 1))
                else:
                    contour_list_tmp.append(self.contour_list[i])
            self.contour_list = contour_list_tmp
            self.contour_use_flag = [False] * len(self.contour_list)
        current_contour_index = 0
        while not self.__is_all_counter_used():
            current_contour = self.contour_list[current_contour_index]
            contour_repeat_num = 0
            contour_len = len(current_contour)
            if contour_len < 5:
                contour_repeat_num = self.tiny_contour_repeat_num
            else:
                contour_repeat_num = 1
            while contour_repeat_num > 0:
                if contour_len == 1:
                    self.__append_point(self.points, self.__to_ilda_point(current_contour[0], True))
                    pass
                else:
                    for j in range(1, len(current_contour)):
                        if j == 1:
                            self.__append_point(self.points, self.__to_ilda_point(current_contour[j - 1], True))
                        points = self.get_points_between(current_contour[j - 1], current_contour[j])
                        self.points += points
                contour_repeat_num -= 1
            # for i in range(5):
            #     self.points.append(self.__to_ilda_point(current_contour[-1], True))
            next_idx = self.__find_nearest_contour(current_contour_index)
            if next_idx is not None:
                for i in range(self.end_blank_repeat_num):
                    self.__append_point(self.points, self.__to_ilda_point(current_contour[-1], False))
                blank_points = self.get_points_between(current_contour[-1], self.contour_list[next_idx][0],
                                                       False)
                self.points += blank_points
                for i in range(self.end_blank_repeat_num):
                    self.__append_point(self.points, self.__to_ilda_point(self.contour_list[next_idx][0], False))
                current_contour_index = next_idx
        # 增加运动点位返回起点
        if self.__is_all_counter_used():
            blank_points = self.get_points_between(self.contour_list[current_contour_index][-1],
                                                   self.contour_list[0][0],
                                                   False)
            self.points += blank_points
        if self.image is not None:
            point_image = np.zeros(self.image.shape)
            for i in self.org_points:
                point_image[i[0]][i[1]] = rgb_to_7_color(self.image[i[0]][i[1]])
                # print(point_image[i[0]][i[1]])
            self.__output_img("point_image", point_image, img_type=".bmp")
        return self.points

    def __find_nearest_contour(self, contour_index):
        min_dis = 99999999.0
        last_point = self.contour_list[contour_index][-1]
        nearest_idx = None
        self.contour_use_flag[contour_index] = True
        for j in range(len(self.contour_list)):
            first_point = self.contour_list[j][0]
            if not self.contour_use_flag[j]:
                dis = math.sqrt(
                    math.pow(last_point[0] - first_point[0], 2) + math.pow(last_point[1] - first_point[1], 2))
                if dis < min_dis:
                    nearest_idx = j
                    min_dis = dis
        # self.contour_use_flag[nearest_idx] = True
        return nearest_idx

    def __is_all_counter_used(self):
        for i in self.contour_use_flag:
            if not i:
                return False
        return True

    def get_points_between(self, point1, point2, enable=True):
        points = []
        dis = math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2)) * self.scales
        if enable:
            middle_points_count = round(dis / self.laser_point_interval) - 1
        else:
            middle_points_count = round(dis / self.blank_point_interval) + 3
        if dis > self.dis_threshold:
            self.__append_point(points, self.__to_ilda_point(point1, enable))
            self.__append_point(points, self.__to_ilda_point(point1, enable))
        if middle_points_count > 0:
            x_interval = (point2[0] - point1[0]) / (middle_points_count + 1)
            y_interval = (point2[1] - point1[1]) / (middle_points_count + 1)
            for i in range(middle_points_count):
                x_middle = point1[0] + x_interval * i
                y_middle = point1[1] + y_interval * i
                if len(point1) > 2:
                    self.__append_point(points, self.__to_ilda_point([x_middle, y_middle, point1[2]], enable))
                else:
                    self.__append_point(points, self.__to_ilda_point([x_middle, y_middle], enable))
        if dis > self.dis_threshold:
            self.__append_point(points, self.__to_ilda_point(point2, enable))
            self.__append_point(points, self.__to_ilda_point(point2, enable))
        self.__append_point(points, self.__to_ilda_point(point2, enable))
        return points

    def __append_point(self, points, point):
        if point is not None:
            points.append(point)
        pass

    def __to_ilda_point(self, point, enable):
        x = int(point[0] * self.scales) - 32767
        #  y 反向
        y = -(int(point[1] * self.scales) - 32767)
        if self.image is not None:
            color = self.image[int(point[1]), int(point[0])]
            color_bgr = [color[2], color[1], color[0]]
            color = rgb_to_ilda_color(color_bgr)
        else:
            if len(point) > 2:
                color = rgb_to_ilda_color(point[2])
            else:
                color = COLOR.WHITE
        if is_out_of_signed_short_range(x) or is_out_of_signed_short_range(y):
            return None
        self.org_points.append([int(point[1]), int(point[0])])
        return [x, y, 0, color, enable]

    def __find_skeleton(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gary", gray_image)
        ret, binary = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow("binary", binary)
        skeleton0 = morphology.skeletonize(binary)
        self.skeleton = skeleton0.astype(np.uint8) * 255
        self.__output_img("skeleton", self.skeleton)

        color_edge_image = cv2.bitwise_and(self.image, self.image, mask=self.skeleton)
        self.__output_img("edge_image", color_edge_image)
        pass

    def __output_img(self, name, img, img_type=".jpg"):
        if self.save_image:
            dir_path = os.path.dirname(os.path.abspath(__file__)) + "\\output\\"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            # print(dir_path + name + ".jpg")
            cv2.imwrite(dir_path + name + img_type, img)
