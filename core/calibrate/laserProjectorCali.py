import cv2
import cv2.aruco as aruco
import numpy

from core.imageConverter import ImageILDAConverter
from core.lidaGenerator import LidaFile
from core.PointAddStrategy import CirclePointStrategy
from core.utils import rotationVectorToEulerAngles
import numpy as np


def point_compare(x, y):
    if x[0] != y[0]:
        return


class LaserProjectorCalibration:
    def __init__(self):
        self.row_num = 5
        self.col_num = 5
        self.off_set_x = 5000
        self.off_set_y = 8000
        self.checker_width = 5000
        self.point_strategy = CirclePointStrategy()
        self.point_strategy.point_size = 2000
        self.image = None
        self.image_list = np.array([])
        self.camera_mtx = None
        self.camera_dist = None
        self.start_cali = False
        self.cali_count = 0
        self.max_image_num = 100
        self.projector_width = 65535
        self.projector_height = 65535
        pass

    def get_one_image(self, row_idx, col_idx):
        return self.__generate_cali_image(row_idx, col_idx)

    def get_projection_image(self):
        lida_file = LidaFile()
        lida_file.name = "cali"
        lida_file.company = "buz141"
        lida_file.new_frame()
        convertor = ImageILDAConverter()
        convertor.scales = 1.0
        contours = []
        for i in range(self.row_num):
            for j in range(self.col_num):
                contours.append(self.__generate_one_point(i, j)[0])
        convertor.contour_list = np.array(contours, dtype=object)
        points = convertor.convert_contour_to_projection()
        for point in points:
            lida_file.add_point(point[0], point[1], point[2], point[3], point[4])
        return lida_file

    def set_param(self, row=5, col=5, checker_width=5000, camera_mtx=None, camera_dist=None):
        self.row_num = row
        self.col_num = col
        self.checker_width = checker_width
        self.camera_mtx = camera_mtx
        self.camera_dist = camera_dist
        pass

    def start_record_cali_img(self):
        self.start_cali = True

    def stop_record_cali_img(self):
        self.start_cali = False
        self.image = None
        self.cali_count = 0

    def is_start(self):
        return self.start_cali

    def add_image(self, image):
        if not self.start_cali:
            return
        if self.cali_count >= self.max_image_num:
            self.start_cali = False
            return
        if self.image is None:
            self.image = image
        else:
            self.image_list = np.array([self.image, image])
            self.image = np.max(self.image_list, axis=0)
            self.cali_count += 1

    def __generate_cali_image(self, row_idx, col_idx):
        lida_file = LidaFile()
        lida_file.name = "cali"
        lida_file.company = "buz141"
        lida_file.new_frame()
        convertor = ImageILDAConverter()
        convertor.scales = 1.0
        convertor.contour_list = np.array([self.__generate_one_point(row_idx, col_idx)[0]], dtype=object)
        points = convertor.convert_contour_to_projection()
        for point in points:
            lida_file.add_point(point[0], point[1], point[2], point[3], point[4])
        return lida_file

    def __generate_one_point(self, row_idx, col_idx):
        x, y = self.off_set_x + self.checker_width * row_idx, self.off_set_y + self.checker_width * col_idx
        contour = self.point_strategy.get_contour(x, y, [255, 255, 255])
        return contour, (x, y)

    def calibrate(self, image):
        h, w = image.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_mtx, self.camera_dist, (w, h), 0, (w, h))  #
        distort_image = cv2.undistort(image, self.camera_mtx, self.camera_dist, None, new_camera_mtx)
        gray_image = cv2.cvtColor(distort_image, cv2.COLOR_BGR2GRAY)
        checker_points = self.find_circle_center(gray_image)
        self.cali_projector(gray_image, checker_points)

    def find_circle_center(self, gray_image):
        ret, white_image = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)
        self.__output_img("white_image", white_image)
        contour_list, hierarchy = cv2.findContours(white_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # contour_image = np.zeros(white_image.shape)
        # new_contour_list = []
        circle_list = []
        checker_points = []
        for i in range(len(contour_list)):
            if hierarchy[0][i][-1] == -1:
                # new_contour_list.append(contour_list[i])
                center, radius = cv2.minEnclosingCircle(contour_list[i])
                circle_list.append([int(center[0]), int(center[1]), int(radius)])
                checker_points.append([center[0], center[1]])
        # contour_image = cv2.drawContours(contour_image, new_contour_list, -1, (175, 0, 175), -1)
        # self.__output_img("counter_img", contour_image)
        # center_image = np.zeros(white_image.shape)
        # # img = cv2.medianBlur(gray_img, 5)
        # for i in circle_list:
        #     # 画出来圆的边界
        #     # cv2.circle(center_image, (i[0], i[1]), i[2], (255, 0, 255), 2)
        #     # 画出来圆心
        #     cv2.circle(center_image, (i[0], i[1]), 2, (255, 255, 255), 2)
        #     cv2.circle(distort_image, (i[0], i[1]), 2, (255, 255, 255), 2)
        # self.__output_img("center_image", center_image)
        # self.__output_img("distort_image", distort_image)
        return checker_points

    def cali_projector(self, gray_image, checker_points):
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image,
                                                              aruco_dict,
                                                              parameters=parameters)

        h, w = gray_image.shape[:2]
        word_point_list = []
        trans_mat = None
        if ids is not None:
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 83.12, self.camera_mtx, self.camera_dist)
            r = np.zeros((3, 3), dtype=np.float64)
            cv2.Rodrigues(rvec[0][0], r)
            print(tvec[0][0])
            trans_mat = np.concatenate((np.concatenate((r, tvec[0][0].reshape(3, 1)), axis=1), [[0, 0, 0, 1]]),
                                       axis=0)
            # self.__draw_origin(gray_image, trans_mat)
            print(trans_mat)
            in_mtx_inv = np.linalg.inv(self.camera_mtx)
            trans_inv = np.linalg.inv(trans_mat)
            krt_inv = np.linalg.inv(np.concatenate((self.camera_mtx.dot(trans_mat[0:3]), [[0, 0, 0, 1]]), axis=0))
            r_inv = np.linalg.inv(r)
            for point in checker_points:
                point.append(1)
                # point.append(1)
                word_point = in_mtx_inv.dot(point)
                # word_point = np.append(word_point, 1)
                # print(word_point)
                # word_point -= tvec[0][0]
                # print(word_point)
                word_point = r_inv.dot(word_point)
                # print("word_point= ", word_point)
                p = tvec[0][0]
                d = r_inv.dot(p)
                # print(n, p)
                # print("d= ", d)
                # print("sw= ", sw)
                s = d[2] / word_point[2]
                # print("s= ", s)
                word_point = word_point * s - d
                word_point_list.append(word_point)
                # print(word_point[0], word_point[1], word_point[2])
        projector_point_list = []

        # 按先行再列的生成顺序，与opencv棋盘格标定时坐标轴顺序一致
        for i in range(self.row_num - 1, -1, -1):
            for j in range(self.col_num):
                projector_point_list.append([self.__generate_one_point(j, i)[1]])

        projector_point_list = np.array(projector_point_list).astype(np.float32)
        # projector_point_list[:, :, 0] -= 32767
        # projector_point_list[:, :, 1] = 32767 - projector_point_list[:, :, 1]
        # projector_point_list[:, :, 1] = 65534 - projector_point_list[:, :, 1]
        projector_point_list = [projector_point_list]
        word_point_list = np.array(word_point_list).astype(np.float32)
        word_point_list = word_point_list[word_point_list[:, 1].argsort()]
        for i in range(self.col_num):
            start, end = i * self.row_num, (i + 1) * self.row_num
            sub_list = word_point_list[start:end]
            word_point_list[start:end] = sub_list[sub_list[:, 0].argsort()]
        # word_point_list[:, 0] -= 5000
        word_point_list = [word_point_list]
        ret, mtx, dist, pro_rvecs, pro_tvecs = cv2.calibrateCamera(word_point_list, projector_point_list,
                                                                   (self.projector_width, self.projector_height),
                                                                   None, None)
        projector_r = np.zeros((3, 3), dtype=np.float64)
        cv2.Rodrigues(pro_rvecs[0], projector_r)
        print(ret)
        print(mtx)
        print(dist)
        print(pro_rvecs)
        print(pro_tvecs)
        print(projector_r)
        trans_mat_pro = np.concatenate(
            (np.concatenate((projector_r, pro_tvecs[0].reshape(3, 1)), axis=1), [[0, 0, 0, 1]]),
            axis=0)
        print("trans_mat_pro = \n", trans_mat_pro)
        print("trans_mat_pro_inv = \n", np.linalg.inv(trans_mat_pro))
        camera_projector_trans_mat = np.dot(trans_mat_pro, np.linalg.inv(trans_mat))
        print("camera_projector_trans_mat =\n ", camera_projector_trans_mat)
        angle = rotationVectorToEulerAngles(pro_rvecs[0])
        print(angle)

    def __draw_origin(self, gray_image, trans_mat):
        origin = np.array([0, 0, 0, 1]).reshape(-1, 1)
        tmp = trans_mat[0:3].dot(origin)
        print("tmp= ", tmp)
        origin = self.camera_mtx.dot(tmp) / trans_mat[2, 3]
        print("origin= ", origin)
        origin = origin.reshape(-1)
        cv2.circle(gray_image, (int(origin[0]), int(origin[1])), 2, (255, 255, 255), 5)
        self.__output_img("origin", gray_image)

    def __output_img(self, name, image):
        cv2.imwrite(name + ".jpg", image)
