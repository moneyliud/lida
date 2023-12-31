import math

from core.Point3DToLida import Point3DToLida
from core.PointAddStrategy import CirclePointStrategy
import numpy as np
import serial
import cv2


def generate_point3d_lida(points, trans_mat, color):
    converter = Point3DToLida()
    converter.camera_project_trans_mtx = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0],
                                                   [0, 0, 1, 0],
                                                   [0, 0, 0, 1]], np.float64)
    image1 = cv2.imread("C:\\Users\\Administrator\\Pictures\\527NKJ84.png")
    image2 = cv2.imread("C:\\Users\\Administrator\\Pictures\\00328.png")
    image3 = cv2.imread("C:\\Users\\Administrator\\Pictures\\506K8.png")
    image_list = [image1, image2, image3, image3]
    converter.convert_type = "origin"
    # converter.new_frame(duration=3)
    # converter.add_image(image1)

    converter.camera_trans_mtx = trans_mat
    converter.set_strategy(CirclePointStrategy())
    for i, point_group in enumerate(points):
        converter.new_frame(duration=3)
        converter.add_image(image_list[i])
        converter.new_frame(duration=3)
        for j, point in enumerate(point_group):
            converter.add_point(point, color[i])
        # converter.new_frame(duration=3)
    # for i in range(len(points)):
    #     converter.add_point(points[i], np.array([0, 255, 0]))
    # converter.new_frame(duration=3)
    #
    # points1 = points[np.where((color == [255, 255, 0]).all(axis=-1))]
    # color1 = color[np.where((color == [255, 255, 0]).all(axis=-1))]
    # for i in range(len(points1)):
    #     converter.add_point(points1[i], color1[i])
    # converter.new_frame(duration=3)
    #
    # points2 = points[np.where((color == [255, 255, 255]).all(axis=-1))]
    # color2 = color[np.where((color == [255, 255, 255]).all(axis=-1))]
    # for i in range(len(points2)):
    #     converter.add_point(points2[i], color2[i])
    # converter.new_frame(duration=3)

    return converter.to_bytes()
    pass


if __name__ == "__main__":
    target_point = np.array([[[1099.311599, 8953, -197.149], [1095.643198, 8953, -127.152],
                              [1096.691784, 8953, -147.16], [1097.740371, 8953, -167.168]],
                             [[1100.780763, 8953, -225.182]],
                             [[1107.41044, 8951, -256.147], [1110.225428, 8960.5, -300.307],
                              [1111.452588, 8949, -323.722], [1112.679748, 8960.5, -347.138],
                              [1113.906908, 8949, -370.554], [1115.134068, 8960.5, -393.969],
                              [1116.361228, 8949, -417.385], [1117.643129, 8960.5, -441.845],
                              [1118.703853, 8949, -462.085], [1119.6459, 8960.5, -480.06],
                              [1120.920701, 8949, -504.385], [1122.195501, 8960.5, -528.709],
                              [1123.470301, 8949, -553.034], [1124.745101, 8960.5, -577.359],
                              [1126.028374, 8949, -601.845], [1126.788502, 8960.5, -616.349],
                              [1127.54863, 8949, -630.853], [1128.253042, 8960.5, -653.848]],
                             [[1126.539384, 8959, -678.471]]], dtype=object)
    color = np.array([[255, 255, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
    trans_mat = np.linalg.inv(np.array([[0, 0, 1, 2749.78],
                                        [0, -1, 0, 7205.31],
                                        [1, 0, 0, -819.484],
                                        [0, 0, 0, 1]]))
    rotate_angle_x = 47.88 / 180 * math.pi
    rotate_angle_y = 10.714 / 180 * math.pi
    rotate_mat_y = np.array([[math.cos(rotate_angle_y), 0, math.sin(rotate_angle_y), 0],
                             [0, 1, 0, 0],
                             [-math.sin(rotate_angle_y), 0, math.cos(rotate_angle_y), 0],
                             [0, 0, 0, 1]])
    rotate_mat_x = np.array([[1, 0, 0, 0],
                             [0, math.cos(rotate_angle_x), -math.sin(rotate_angle_x), 0],
                             [0, math.sin(rotate_angle_x), math.cos(rotate_angle_x), 0],
                             [0, 0, 0, 1]])
    trans_mat = np.dot(rotate_mat_x, trans_mat)
    trans_mat = np.dot(rotate_mat_y, trans_mat)
    for i, point_group in enumerate(target_point):
        b = [[1.0]] * len(point_group)
        target_point[i] = np.concatenate((point_group, b), 1)
    file_bytes = generate_point3d_lida(target_point, trans_mat, color)
    file = open("../product.ild", 'wb')
    file.write(file_bytes)

    file_len = len(file_bytes)
    ser = serial.Serial("COM3", 115200)
    buffer_size = 2048
    if ser.isOpen():
        print(ser.name)
    ser.write(bytes([0xAA, 0xBB]))
    ser.write(file_len.to_bytes(4, "little"))
    write_len = 0
    while write_len < file_len:
        end = (write_len + buffer_size) if write_len + buffer_size < file_len else file_len
        ser.write(file_bytes[write_len:end])
        print(end)
        write_len += buffer_size
    ser.flush()
    pass
