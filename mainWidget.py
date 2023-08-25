# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1582, 802)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.selectImage = QtWidgets.QPushButton(self.centralwidget)
        self.selectImage.setGeometry(QtCore.QRect(10, 540, 75, 23))
        self.selectImage.setObjectName("selectImage")
        self.sendImage = QtWidgets.QPushButton(self.centralwidget)
        self.sendImage.setGeometry(QtCore.QRect(90, 540, 75, 23))
        self.sendImage.setObjectName("sendImage")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(10, 10, 512, 512))
        self.widget.setObjectName("widget")
        self.imageLabel = QtWidgets.QLabel(self.widget)
        self.imageLabel.setGeometry(QtCore.QRect(10, 10, 512, 512))
        self.imageLabel.setAutoFillBackground(True)
        self.imageLabel.setObjectName("imageLabel")
        self.cameraLabel = QtWidgets.QLabel(self.centralwidget)
        self.cameraLabel.setGeometry(QtCore.QRect(540, 10, 1000, 750))
        self.cameraLabel.setObjectName("cameraLabel")
        self.wholeBoundary = QtWidgets.QPushButton(self.centralwidget)
        self.wholeBoundary.setGeometry(QtCore.QRect(70, 610, 51, 23))
        self.wholeBoundary.setObjectName("wholeBoundary")
        self.leftBoundary = QtWidgets.QPushButton(self.centralwidget)
        self.leftBoundary.setGeometry(QtCore.QRect(10, 610, 51, 23))
        self.leftBoundary.setObjectName("leftBoundary")
        self.rightBoundary = QtWidgets.QPushButton(self.centralwidget)
        self.rightBoundary.setGeometry(QtCore.QRect(130, 610, 51, 23))
        self.rightBoundary.setObjectName("rightBoundary")
        self.upBoundary = QtWidgets.QPushButton(self.centralwidget)
        self.upBoundary.setGeometry(QtCore.QRect(70, 580, 51, 23))
        self.upBoundary.setObjectName("upBoundary")
        self.downBoundary = QtWidgets.QPushButton(self.centralwidget)
        self.downBoundary.setGeometry(QtCore.QRect(70, 640, 51, 23))
        self.downBoundary.setObjectName("downBoundary")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 670, 461, 101))
        self.groupBox.setObjectName("groupBox")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(10, 20, 41, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(120, 20, 41, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(210, 20, 41, 16))
        self.label_3.setObjectName("label_3")
        self.alpha_label = QtWidgets.QLabel(self.groupBox)
        self.alpha_label.setGeometry(QtCore.QRect(50, 20, 54, 20))
        self.alpha_label.setText("")
        self.alpha_label.setObjectName("alpha_label")
        self.beta_label = QtWidgets.QLabel(self.groupBox)
        self.beta_label.setGeometry(QtCore.QRect(150, 20, 54, 20))
        self.beta_label.setText("")
        self.beta_label.setObjectName("beta_label")
        self.gama_label = QtWidgets.QLabel(self.groupBox)
        self.gama_label.setGeometry(QtCore.QRect(240, 20, 54, 20))
        self.gama_label.setText("")
        self.gama_label.setObjectName("gama_label")
        self.y_label = QtWidgets.QLabel(self.groupBox)
        self.y_label.setGeometry(QtCore.QRect(150, 50, 54, 20))
        self.y_label.setText("")
        self.y_label.setObjectName("y_label")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(30, 50, 16, 16))
        self.label_4.setObjectName("label_4")
        self.x_label = QtWidgets.QLabel(self.groupBox)
        self.x_label.setGeometry(QtCore.QRect(50, 50, 54, 20))
        self.x_label.setText("")
        self.x_label.setObjectName("x_label")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(130, 50, 21, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setGeometry(QtCore.QRect(220, 50, 16, 16))
        self.label_6.setObjectName("label_6")
        self.z_label = QtWidgets.QLabel(self.groupBox)
        self.z_label.setGeometry(QtCore.QRect(240, 50, 54, 20))
        self.z_label.setText("")
        self.z_label.setObjectName("z_label")
        self.y_label_2 = QtWidgets.QLabel(self.groupBox)
        self.y_label_2.setGeometry(QtCore.QRect(150, 80, 54, 20))
        self.y_label_2.setText("")
        self.y_label_2.setObjectName("y_label_2")
        self.label_9 = QtWidgets.QLabel(self.groupBox)
        self.label_9.setGeometry(QtCore.QRect(220, 80, 16, 16))
        self.label_9.setObjectName("label_9")
        self.z_label_2 = QtWidgets.QLabel(self.groupBox)
        self.z_label_2.setGeometry(QtCore.QRect(240, 80, 54, 20))
        self.z_label_2.setText("")
        self.z_label_2.setObjectName("z_label_2")
        self.label_8 = QtWidgets.QLabel(self.groupBox)
        self.label_8.setGeometry(QtCore.QRect(130, 80, 21, 16))
        self.label_8.setObjectName("label_8")
        self.x_label_2 = QtWidgets.QLabel(self.groupBox)
        self.x_label_2.setGeometry(QtCore.QRect(50, 80, 54, 20))
        self.x_label_2.setText("")
        self.x_label_2.setObjectName("x_label_2")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setGeometry(QtCore.QRect(30, 80, 16, 16))
        self.label_7.setObjectName("label_7")
        self.findTarget = QtWidgets.QPushButton(self.centralwidget)
        self.findTarget.setGeometry(QtCore.QRect(260, 610, 75, 23))
        self.findTarget.setObjectName("findTarget")
        self.projectorCaliBtn = QtWidgets.QPushButton(self.centralwidget)
        self.projectorCaliBtn.setGeometry(QtCore.QRect(260, 580, 75, 23))
        self.projectorCaliBtn.setObjectName("projectorCaliBtn")
        self.projectorCameraCaliBtn = QtWidgets.QPushButton(self.centralwidget)
        self.projectorCameraCaliBtn.setGeometry(QtCore.QRect(260, 550, 75, 23))
        self.projectorCameraCaliBtn.setObjectName("projectorCameraCaliBtn")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.selectImage.setText(_translate("MainWindow", "选择图像"))
        self.sendImage.setText(_translate("MainWindow", "发送图像"))
        self.imageLabel.setText(_translate("MainWindow", "请选择图像"))
        self.cameraLabel.setText(_translate("MainWindow", "实时图像"))
        self.wholeBoundary.setText(_translate("MainWindow", "全边界"))
        self.leftBoundary.setText(_translate("MainWindow", "左边界"))
        self.rightBoundary.setText(_translate("MainWindow", "右边界"))
        self.upBoundary.setText(_translate("MainWindow", "上边界"))
        self.downBoundary.setText(_translate("MainWindow", "下边界"))
        self.groupBox.setTitle(_translate("MainWindow", "相机位姿"))
        self.label.setText(_translate("MainWindow", "alpha:"))
        self.label_2.setText(_translate("MainWindow", "beta:"))
        self.label_3.setText(_translate("MainWindow", "gama:"))
        self.label_4.setText(_translate("MainWindow", "X:"))
        self.label_5.setText(_translate("MainWindow", "Y:"))
        self.label_6.setText(_translate("MainWindow", "Z:"))
        self.label_9.setText(_translate("MainWindow", "Z:"))
        self.label_8.setText(_translate("MainWindow", "Y:"))
        self.label_7.setText(_translate("MainWindow", "X:"))
        self.findTarget.setText(_translate("MainWindow", "自动配准"))
        self.projectorCaliBtn.setText(_translate("MainWindow", "投影仪标定"))
        self.projectorCameraCaliBtn.setText(_translate("MainWindow", "外参标定"))
