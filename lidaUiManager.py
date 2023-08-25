# -*- coding: utf-8 -*-

import cv2
import cv2.aruco as aruco
import os
import time
from pathlib import Path
import serial
import numpy as np
import time
import threading
from mainWidget import Ui_MainWindow
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt, QObject
from core.lidaGenerator import LidaFile
from core.imageConverter import ImageILDAConverter
from core.LIDA import COLOR
from core.Point3DToLida import Point3DToLida
from core.calibrate.laserProjectorCali import LaserProjectorCalibration
from core.utils import rotationVectorToEulerAngles
from core.PointAddStrategy import SinglePointStrategy


class LidaUiManager(QObject):
    def __init__(self, ui: Ui_MainWindow, main_window: QMainWindow):
        super().__init__()
        self.ui = ui
        self.main_window = main_window
        self.image_bytes = None
        self.serial_connect = None
        self.camera = None
        self.frame = None
        self.is_camera_opened = False  # 摄像头有没有打开标记
        self.aruco_image_len = 0.08132 * 1000  # 81.32 mm
        self.projector_in_mtx = np.array([[9.60864149e+04, 0.00000000e+00, 4.69182525e+04],
                                          [0.00000000e+00, 1.08873955e+05, 5.81167521e+04],
                                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        self.mtx = np.array([[1.79697103e+03, 0.00000000e+00, 1.16728650e+03],
                             [0.00000000e+00, 1.79632799e+03, 8.53307321e+02],
                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        self.distort = np.array([[0.05228516, -0.23618041, 0.00102657, 0.0011332, 0.33239178]])
        self.camera_project_trans_mtx = np.array([[9.99023359e-01, -1.82391259e-02, -4.02450181e-02, -2.00164434e+02],
                                                  [1.33238194e-02, 9.92782438e-01, -1.19186855e-01, -2.07905788e+02],
                                                  [4.21284113e-02, 1.18534235e-01, 9.92055861e-01, 7.77732594e+00],
                                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                                                 np.float64)
        # self.distort = np.array([[0.0001, -0.0001, 0.00098506, 0.000112824, 0.00001]])
        self.target_point = np.array(
            [[230.0, 0, 0, 1.0], [230.0, -30, 0, 1.0], [230.0, 30, 0, 1.0], [100.0, 0, 0, 1.0], [100.0, -30, 0, 1.0],
             [100.0, 30, 0, 1.0]])
        self.target_contour = np.array(
            [[-50, 50, 0, 1.0], [50, 50, 0, 1.0], [50, -50, 0, 1.0], [-50, -50, 0, 1.0],
             [-50, 50, 0, 1.0]])
        # self.target_point = np.array(
        #     [[0, 0, 0, 1.0]])
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters()

        self.camera_trans_mtx = np.array([[1, 0, 0, 0],
                                          [0, 1, 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]], np.float64)
        self.calibrator = LaserProjectorCalibration()
        self.min_send_interval = 1
        self.last_send_time = 0
        self._timer = QTimer(self)
        self._cali_timer = QTimer(self)
        self.__init_ui()
        self.__connect_equip()
        self.__init_camera()

    def __connect_equip(self):
        self.serial_connect = serial.Serial("COM3", 115200)

    def __init_ui(self):
        self.ui.selectImage.clicked.connect(self.__select_image)
        self.ui.sendImage.clicked.connect(self.__send_image)
        self.ui.upBoundary.clicked.connect(self.__project_boundary)
        self.ui.leftBoundary.clicked.connect(self.__project_boundary)
        self.ui.rightBoundary.clicked.connect(self.__project_boundary)
        self.ui.downBoundary.clicked.connect(self.__project_boundary)
        self.ui.wholeBoundary.clicked.connect(self.__project_boundary)
        self.ui.findTarget.clicked.connect(self.__project_target_image)
        self.ui.projectorCaliBtn.clicked.connect(self.__projector_calibration)
        self.ui.projectorCameraCaliBtn.clicked.connect(self.__projector_camera_calibration)

    def __init_camera(self):
        self.camera = cv2.VideoCapture(0)
        # self.camera = cv2.VideoCapture("C:\\Users\\Administrator\\Pictures\\WeChat_20230705205725.mp4")
        # 定时器：30ms捕获一帧
        self._timer.timeout.connect(self.__get_camera_image)
        self._timer.setInterval(30)
        self._timer.start()

    def __select_image(self):
        file_name = QFileDialog.getOpenFileNames(self.main_window, '选择图像', os.getcwd(),
                                                 "Image files (*.jpg *.gif *.png *.jpeg)")
        if len(file_name[0]) > 0:
            pix_map = QPixmap(file_name[0][0])
            self.ui.imageLabel.setPixmap(pix_map)
            self.ui.imageLabel.setScaledContents(True)
            self.__generate_lida_image(file_name[0][0])

    def __generate_lida_image(self, file_name):
        lida_file = LidaFile()
        lida_file.name = Path(file_name).stem
        lida_file.company = "buz141"
        image = cv2.imread(file_name)
        converter = ImageILDAConverter()
        converter.save_image = True
        points = converter.convert(image)
        lida_file.new_frame()
        print(len(points))
        for i in range(len(points)):
            point = points[i]
            lida_file.add_point(point[0], point[1], point[2], point[3], point[4])
        self.image_bytes = lida_file.to_bytes()

    def __send_image(self):
        if self.image_bytes is None:
            reply = QMessageBox.information(self.main_window, '消息', '请选择图像', QMessageBox.Ok,
                                            QMessageBox.Ok)
            return
        self.__send_bytes(self.image_bytes)

    def __get_camera_image(self):
        ret, self.frame = self.camera.read()
        img_rows, img_cols, channels = self.frame.shape
        bytesPerLine = channels * img_cols
        self.__process_frame()
        cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB, self.frame)
        QImg = QImage(self.frame.data, img_cols, img_rows, bytesPerLine, QImage.Format_RGB888)
        self.ui.cameraLabel.setPixmap(
            QPixmap.fromImage(QImg).scaled(self.ui.cameraLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        pass

    def __process_frame(self):
        gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        if self.calibrator.start_cali:
            self.calibrator.add_image(gray_frame)
        '''
        detectMarkers(...)
            detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
            mgPoints]]]]) -> corners, ids, rejectedImgPoints
        '''

        # lists of ids and the corners beloning to each id
        # detecter = aruco.ArucoDetector()
        # detecter.setDetectorParameters(parameters)
        # detecter.setDictionary(aruco_dict)
        # corners, ids, rejectedImgPoints = detecter.detectMarkers(gray_frame)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_frame,
                                                              self.aruco_dict,
                                                              parameters=self.parameters)
        obj_points = np.array([[0, 0, 0], [0, self.aruco_image_len, 0], [self.aruco_image_len, self.aruco_image_len, 0],
                               [self.aruco_image_len, 0, 0]], dtype=np.float32)

        h, w = self.frame.shape[:2]
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.distort, (w, h), 0, (w, h))  #
        if ids is not None:
            rvecs_list = []
            tvecs_list = []
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 83.12, self.mtx, self.distort)
            trans_mat = None
            for i in range(rvec.shape[0]):
                cv2.drawFrameAxes(self.frame, self.mtx, self.distort, rvec[i, :, :], tvec[i, :, :], 80)
                aruco.drawDetectedMarkers(self.frame, corners)
                if ids[i][0] == 6:
                    r = np.zeros((3, 3), dtype=np.float64)
                    cv2.Rodrigues(rvec[i][0], r)
                    trans_mat = np.concatenate((np.concatenate((r, tvec[i][0].reshape(3, 1)), axis=1), [[0, 0, 0, 1]]),
                                               axis=0)
                    self.camera_trans_mtx = trans_mat
                    tp = np.dot(trans_mat, self.target_point[0])
                    self.ui.x_label_2.setText(str(tp[0]))
                    self.ui.y_label_2.setText(str(tp[1]))
                    self.ui.z_label_2.setText(str(tp[2]))
                    # print(trans_mat, inv_mat, ids[i])
                    # print(tp)
            # for corner in corners:\
            #     success, rvecs, tvecs, = cv2.solvePnP(obj_points, corner, self.mtx, self.distort)
            #     rvecs_list.append(rvecs)
            #     tvecs_list.append(tvecs)
            #     cv2.drawFrameAxes(self.frame, self.mtx, self.distort, rvecs, tvecs, 300)
            # aruco.drawAxis(self.frame, self.mtx, self.distort, rvecs, tvecs, 0.1)  # Draw Axis
            # aruco.drawDetectedMarkers(self.frame, corners)
            # self.__send_bytes(self.generate_point3d_lida(self.target_point, self.camera_trans_mtx))
            h, w = self.frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.distort, (w, h), 0, (w, h))
            self.frame = cv2.undistort(self.frame, self.mtx, self.distort, None, newcameramtx)
            r = rotationVectorToEulerAngles(rvec[0])
            rvec, tvec = self.__calculate_rtvec(rvec, tvec)
            self.ui.alpha_label.setText(str(r[0]))
            self.ui.beta_label.setText(str(r[1]))
            self.ui.gama_label.setText(str(r[2]))
            self.ui.x_label.setText(str(tvec[0][0]))
            self.ui.y_label.setText(str(tvec[0][1]))
            self.ui.z_label.setText(str(tvec[0][2]))
        pass

    def __project_target_image(self):
        # file_bytes = self.generate_point3d_lida(self.target_point, self.camera_trans_mtx, self.target_contour)
        file_bytes = self.generate_point3d_lida(None, self.camera_trans_mtx, self.target_contour)
        file = open("target.ild", 'wb')
        file.write(file_bytes)
        self.__send_bytes(file_bytes)
        pass

    def generate_point3d_lida(self, points, trans_mat, contour=None):
        converter = Point3DToLida()
        converter.projector_in_mtx = self.projector_in_mtx
        converter.camera_project_trans_mtx = self.camera_project_trans_mtx
        converter.camera_trans_mtx = trans_mat
        converter.set_strategy(SinglePointStrategy())
        if points is not None:
            for point in points:
                converter.add_point(point)
            print(len(points))
        if contour is not None:
            converter.add_contour(contour, [255, 255, 255])
        converter.new_frame(0)
        return converter.to_bytes()
        pass

    def __calculate_rtvec(self, rvecs_list, tvecs_list):
        if len(rvecs_list) > 0 and len(tvecs_list) > 0:
            return rvecs_list[0], tvecs_list[0]
        return None, None

    def __project_boundary(self):
        p_button = self.sender()
        lida_file = LidaFile()
        lida_file.name = "boundary"
        lida_file.company = "buz141"
        lida_file.new_frame()
        points = []
        converter = ImageILDAConverter()
        converter.scales = 1
        converter.laser_point_interval = 500
        # max_boundary = 60000
        # min_boundary = 5000
        max_boundary = 65534
        min_boundary = 0
        if "up" in p_button.objectName():
            points = [[min_boundary, min_boundary], [max_boundary, min_boundary]]
        elif "left" in p_button.objectName():
            points = [[min_boundary, min_boundary], [min_boundary, max_boundary]]
            pass
        elif "right" in p_button.objectName():
            points = [[max_boundary, min_boundary], [max_boundary, max_boundary]]
            pass
        elif "down" in p_button.objectName():
            points = [[min_boundary, max_boundary], [max_boundary, max_boundary]]
            pass
        elif "whole" in p_button.objectName():
            points = [[min_boundary, min_boundary], [max_boundary, min_boundary], [max_boundary, max_boundary],
                      [min_boundary, max_boundary], [min_boundary, min_boundary]]
            pass
        for i in range(1, len(points)):
            ret = converter.get_points_between(points[i - 1], points[i])
            for j in ret:
                lida_file.add_point(j[0], j[1], 0, COLOR.WHITE)
        ret = converter.get_points_between(points[len(points) - 1], points[0])
        for j in ret:
            lida_file.add_point(j[0], j[1], 0, COLOR.WHITE)
        self.__send_bytes(lida_file.to_bytes())
        pass

    def __save_camera_cali_image(self):
        if self.calibrator.is_start():
            cv2.imwrite("cali.jpg", self.calibrator.image)
            self.calibrator.stop_record_cali_img()
        self._cali_timer.stop()
        pass

    def __projector_calibration(self):
        self.calibrator.set_param(7, 7)
        image_file = self.calibrator.get_projection_image()
        self.__send_bytes(image_file.to_bytes())
        self.calibrator.start_record_cali_img()
        time.sleep(1)
        self._cali_timer.timeout.connect(self.__save_camera_cali_image)
        self._cali_timer.setInterval(5000)
        self._cali_timer.start()
        pass

    def __projector_camera_calibration(self):
        pass

    def __send_bytes(self, image_bytes):
        file_len = len(image_bytes)
        buffer_size = 2048
        interval = time.time() - self.last_send_time
        print("send")
        if self.serial_connect.isOpen() and interval > self.min_send_interval:
            self.serial_connect.write(bytes([0xAA, 0xBB]))
            self.serial_connect.write(file_len.to_bytes(4, "little"))
            write_len = 0
            while write_len < file_len:
                end = (write_len + buffer_size) if write_len + buffer_size < file_len else file_len
                self.serial_connect.write(image_bytes[write_len:end])
                # print(end)
                write_len += buffer_size
            self.serial_connect.flush()
            self.last_send_time = time.time()
