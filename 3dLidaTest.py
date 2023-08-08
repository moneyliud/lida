import math

from core.Point3DToLida import Point3DToLida
import numpy as np
import serial


def generate_point3d_lida(points, trans_mat, color):
    converter = Point3DToLida()
    converter.camera_project_trans_mtx = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0],
                                                   [0, 0, 1, 0],
                                                   [0, 0, 0, 1]], np.float64)
    converter.camera_trans_mtx = trans_mat
    for i in range(len(points)):
        converter.add_point(points[i], color[i])
    return converter.to_bytes()
    pass


if __name__ == "__main__":
    target_point = np.array([
        [552, 13134.5, 1567.599], [552, 13134.5, 1540.034], [552, 13134.5, 1507.469], [552, 13134.5, 1473.904],
        [552, 13134.5, 1441.339], [552, 13134.5, 1406.774], [552, 13134.5, 1373.209], [552, 13134.5, 1339.644],
        [552, 13134.5, 1306.079], [552, 13134.5, 1272.514], [552, 13134.5, 1243.949],
        [552, 13098.7, 1247.449], [552, 13098.7, 1278.497], [552, 13098.7, 1312.045], [552, 13098.7, 1345.512],
        [552, 13098.7, 1379.14], [552, 13098.7, 1412.687], [552, 13098.7, 1448.235], [552, 13098.7, 1479.782],
        [552, 13098.7, 1513.33], [552, 13098.7, 1545.877], [552, 13098.7, 1573.425]])
    color = [[255, 255, 0], [255, 255, 0],
             [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
             [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
             [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
             [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
             [255, 255, 255], [255, 255, 255], [255, 255, 0], [255, 255, 0]]
    trans_mat = np.linalg.inv(np.array([[0, 0, 1, -148],
                                        [0, -1, 0, 12846.6],
                                        [1, 0, 0, 1408.687],
                                        [0, 0, 0, 1]]))
    rotate_angle = -21 / 180 * math.pi
    rotate_mat_y = np.array([[math.cos(rotate_angle), 0, math.sin(rotate_angle), 0],
                             [0, 1, 0, 0],
                             [-math.sin(rotate_angle), 0, math.cos(rotate_angle), 0],
                             [0, 0, 0, 1]])
    rotate_mat_x = np.array([[1, 0, 0, 0],
                             [0, math.cos(rotate_angle), -math.sin(rotate_angle), 0],
                             [0, math.sin(rotate_angle), math.cos(rotate_angle), 0],
                             [0, 0, 0, 1]])
    trans_mat = np.dot(rotate_mat_x, trans_mat)
    b = [[1.0]] * len(target_point)
    target_point = np.concatenate((target_point, b), 1)
    file_bytes = generate_point3d_lida(target_point, trans_mat, color)
    file = open("./product.ild", 'wb')
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
