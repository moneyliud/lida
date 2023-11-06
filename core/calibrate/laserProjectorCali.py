import cv2
import numpy

from core.imageConverter import ImageILDAConverter
from core.lidaGenerator import LidaFile
from core.PointAddStrategy import CirclePointStrategy
from core.utils import rotationVectorToEulerAngles, is_contour_circle
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
        self.xy_num = 4
        self.idx_x = 0
        self.idx_y = 0
        self.image_idx = 0
        self.cali_img_interval = 10000
        self.has_next_image = True
        self.base_image = None
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
        if self.idx_x + 1 < self.xy_num:
            self.idx_x += 1
        else:
            self.idx_x = 0
            if self.idx_y + 1 < self.xy_num:
                self.idx_y += 1
            else:
                self.has_next_image = False
        self.image_idx += 1
        return lida_file

    def init(self, row=5, col=5, checker_width=5000, camera_mtx=None, camera_dist=None, offset_x=5000,
             offset_y=8000, xy_num=4, cali_img_interval=10000):
        self.row_num = row
        self.col_num = col
        self.checker_width = checker_width
        self.camera_mtx = camera_mtx
        self.camera_dist = camera_dist
        self.off_set_x = offset_x
        self.off_set_y = offset_y
        self.xy_num = xy_num
        self.idx_x = 0
        self.idx_y = 0
        self.cali_img_interval = cali_img_interval
        self.image_idx = 0
        self.has_next_image = True
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
        # self.__output_img("org_img" + str(self.cali_count), image)
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
        offset_x = self.off_set_x + self.idx_x * self.cali_img_interval
        offset_y = self.off_set_y + self.idx_y * self.cali_img_interval
        x, y = offset_x + self.checker_width * row_idx, offset_y + self.checker_width * col_idx
        contour = self.point_strategy.get_contour(x, y, [255, 255, 255])
        return contour, (x, y)

    def calibrate(self, images, image_indexes):
        checker_point_list = []
        gray_image_list = []
        self.base_image = cv2.cvtColor(self.base_image, cv2.COLOR_BGR2GRAY)
        for image in images:
            h, w = image.shape[:2]
            new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_mtx, self.camera_dist, (w, h), 0, (w, h))  #
            # distort_image = cv2.undistort(image, self.camera_mtx, self.camera_dist, None, new_camera_mtx)
            # gray_image = cv2.cvtColor(distort_image, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            checker_points = self.find_circle_center(gray_image, self.row_num, self.col_num)
            checker_point_list.append(checker_points)
            gray_image_list.append(gray_image)
        self.cali_projector(gray_image_list, checker_point_list, image_indexes)

    def find_circle_center(self, gray_image, row_num, col_num, threshold=235):
        ret, white_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        self.__output_img("white_image", white_image)
        contour_list, hierarchy = cv2.findContours(white_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = np.zeros(white_image.shape)
        new_contour_list = []
        circle_list = []
        checker_points = []
        for i in range(len(contour_list)):
            # 边缘长度>10排除噪声点
            if hierarchy[0][i][-1] == -1 and len(contour_list[i]) > 20 and is_contour_circle(contour_list[i]):
                new_contour_list.append(contour_list[i])
                center, radius = cv2.minEnclosingCircle(contour_list[i])
                circle_list.append([int(center[0]), int(center[1]), int(radius)])
                checker_points.append([center[0], center[1]])
        contour_image = cv2.drawContours(contour_image, new_contour_list, -1, (175, 0, 175), 0)
        self.__output_img("counter_img", contour_image)
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
        # x从小到大，y从小到大，顺序不能错，重要！
        checker_points = np.array(checker_points)
        checker_points = checker_points[checker_points[:, 1].argsort()]
        for i in range(col_num):
            start, end = i * row_num, (i + 1) * row_num
            sub_list = checker_points[start:end]
            checker_points[start:end] = sub_list[sub_list[:, 0].argsort()]
        return checker_points

    def cali_projector(self, gray_image_list, checker_points, image_indexes):
        checker_board_points = []
        checker_word_points = []
        # 棋盘格模板规格
        w = 11
        h = 9
        objp = np.zeros((w * h, 3), np.float32)
        real_pos = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        real_pos[:, 0] *= 95
        real_pos[:, 1] *= 95
        objp[:, :2] = real_pos
        gray_img_shape = None
        i = 0
        rvecs = []
        tvecs = []
        for gray_image in gray_image_list:
            self.__output_img("gray_img" + str(i), gray_image)
            i += 1
            # h, w = gray_image.shape[:2]
            gray_img_shape = gray_image.shape
            base_image = gray_image
            if self.base_image is not None:
                base_image = self.base_image
            ret, corners = cv2.findChessboardCorners(base_image, (w, h), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
            if corners is None:
                corners = self.find_circle_center(base_image, h, w, threshold=70)
                corners = np.expand_dims(corners, 1).astype(np.float32)
                if len(corners) > 0:
                    ret = True
            if ret:
                checker_word_points.append(objp)
                corners = corners.reshape(-1, 2)
                corners = corners[corners[:, 1].argsort()]
                for i in range(h):
                    start, end = i * w, (i + 1) * w
                    sub_list = corners[start:end]
                    corners[start:end] = sub_list[sub_list[:, 0].argsort()]
                corners = np.expand_dims(corners, 1)
                checker_board_points.append(corners)
                success, rvec, tvec, = cv2.solvePnP(objp, corners, self.camera_mtx, self.camera_dist)
                rvecs.append(rvec)
                tvecs.append(tvec)
                # print(corners)
        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(checker_word_points, checker_board_points,
        #                                                    gray_img_shape[::-1], None, None)
        # 反投影误差
        total_error = 0
        for i in range(len(checker_word_points)):
            imgpoints2, _ = cv2.projectPoints(checker_word_points[i], rvecs[i], tvecs[i], self.camera_mtx,
                                              self.camera_dist)
            error = cv2.norm(checker_board_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            # print(i, error)
            total_error += error
        print("camera total error: ", total_error / len(checker_board_points))
        word_point_list = []
        projector_point_list = []
        trans_mat_list = []
        for i in range(len(rvecs)):
            cur_word_point = []
            rvec = rvecs[i]
            tvec = tvecs[i]
            r = np.zeros((3, 3), dtype=np.float64)
            cv2.Rodrigues(rvec, r)
            print(tvec)
            trans_mat = np.concatenate((np.concatenate((r, tvec), axis=1), [[0, 0, 0, 1]]),
                                       axis=0)
            # self.__draw_origin(gray_image, trans_mat)
            print(trans_mat)
            trans_mat_list.append(trans_mat)
            in_mtx_inv = np.linalg.inv(self.camera_mtx)
            trans_inv = np.linalg.inv(trans_mat)
            print("trans_inv=", trans_inv)
            krt_inv = np.linalg.inv(np.concatenate((self.camera_mtx.dot(trans_mat[0:3]), [[0, 0, 0, 1]]), axis=0))
            r_inv = np.linalg.inv(r)
            for point in checker_points[i]:
                point = np.append(point, 1)
                # point.append(1)
                word_point = in_mtx_inv.dot(point)
                # word_point = np.append(word_point, 1)
                # print(word_point)
                # word_point -= tvec[0][0]
                # print(word_point)
                word_point = r_inv.dot(word_point)
                # print("word_point= ", word_point)
                p = tvec.reshape(-1)
                d = r_inv.dot(p)
                # print(n, p)
                # print("d= ", d)
                # print("sw= ", sw)
                s = d[2] / word_point[2]
                # print("s= ", s)
                word_point = word_point * s - d
                cur_word_point.append(word_point)
                pass
                # print(word_point[0], word_point[1], word_point[2])

            cur_projector_point_list = []

            # 按先行再列的生成顺序，与opencv棋盘格标定时坐标轴顺序一致
            self.idx_x, self.idx_y = image_indexes[i]
            # for j in range(self.row_num - 1, -1, -1):
            for j in range(self.row_num):
                for k in range(self.col_num):
                    cur_projector_point_list.append([self.__generate_one_point(k, j)[1]])

            cur_projector_point_list = np.array(cur_projector_point_list).astype(np.float32)
            # projector_point_list[:, :, 0] -= 32767
            # projector_point_list[:, :, 1] = 32767 - projector_point_list[:, :, 1]
            # projector_point_list[:, :, 1] = 65534 - projector_point_list[:, :, 1]
            projector_point_list.append(cur_projector_point_list)
            cur_word_point = np.array(cur_word_point).astype(np.float32)
            cur_word_point = cur_word_point[cur_word_point[:, 1].argsort()]
            for i in range(self.col_num):
                start, end = i * self.row_num, (i + 1) * self.row_num
                sub_list = cur_word_point[start:end]
                cur_word_point[start:end] = sub_list[sub_list[:, 0].argsort()]
            # word_point_list[:, 0] -= 5000
            word_point_list.append(cur_word_point)
            pass
        ret, mtx, dist, pro_rvecs, pro_tvecs = cv2.calibrateCamera(word_point_list, projector_point_list,
                                                                   (self.projector_width, self.projector_height),
                                                                   None, None)
        projector_r = np.zeros((3, 3), dtype=np.float64)
        for i in range(len(pro_rvecs)):
            cv2.Rodrigues(pro_rvecs[i], projector_r)
            print(ret)
            print(mtx)
            print(dist)
            print("pro_rvecs=", pro_rvecs[i])
            print("pro_tvecs=", pro_tvecs[i])
            # print(projector_r)
            trans_mat_pro = np.concatenate(
                (np.concatenate((projector_r, pro_tvecs[i].reshape(3, 1)), axis=1), [[0, 0, 0, 1]]),
                axis=0)
            # print("trans_mat_pro = \n", trans_mat_pro)
            # print("trans_mat_pro_inv = \n", np.linalg.inv(trans_mat_pro))
            camera_projector_trans_mat = np.dot(trans_mat_pro, np.linalg.inv(trans_mat_list[i]))
            print("camera_projector_trans_mat =\n ", camera_projector_trans_mat)
            angle = rotationVectorToEulerAngles(pro_rvecs[i])
            # print(angle)

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
