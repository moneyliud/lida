from core.LIDA import COLOR
import numpy as np
import cv2
import math


def rotationVectorToEulerAngles(rvec):
    R = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(rvec, R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:  # Æ«º½£¬¸©Ñö£¬¹ö¶¯
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    # Æ«º½£¬¸©Ñö£¬¹ö¶¯»»³É½Ç¶È
    rx = x * 180.0 / 3.141592653589793
    ry = y * 180.0 / 3.141592653589793
    rz = z * 180.0 / 3.141592653589793
    return rx, ry, rz


def rgb_to_ilda_color(hex_color):
    if type(hex_color) == "string":
        rgb_color = hex_to_bit_color(hex_color)
    else:
        rgb_color = rgb_to_bit_color(hex_color)
    color = 0
    if rgb_color[0] == 1 and rgb_color[1] == 0 and rgb_color[2] == 0:  # RED
        color = COLOR.RED
    elif rgb_color[0] == 1 and rgb_color[1] == 1 and rgb_color[2] == 0:  # YELLOW
        color = COLOR.YELLOW
    elif rgb_color[0] == 0 and rgb_color[1] == 1 and rgb_color[2] == 0:  # GREEN
        color = COLOR.GREEN
    elif rgb_color[0] == 0 and rgb_color[1] == 1 and rgb_color[2] == 1:  # CYAN
        color = COLOR.CYAN
    elif rgb_color[0] == 0 and rgb_color[1] == 0 and rgb_color[2] == 1:  # BLUE
        color = COLOR.BLUE
    elif rgb_color[0] == 1 and rgb_color[1] == 0 and rgb_color[2] == 1:  # MAGENTA
        color = COLOR.MAGENTA
    elif rgb_color[0] == 1 and rgb_color[1] == 1 and rgb_color[2] == 1:  # WHITE
        color = COLOR.WHITE
    return color


def rgb_to_bit_color(color_rbg):
    r1 = 1 if color_rbg[0] > 128 else 0
    g1 = 1 if color_rbg[1] > 128 else 0
    b1 = 1 if color_rbg[2] > 128 else 0
    return [r1, g1, b1]


def rgb_to_7_color(color_rbg):
    r1 = 255 if color_rbg[0] > 128 else 0
    g1 = 255 if color_rbg[1] > 128 else 0
    b1 = 255 if color_rbg[2] > 128 else 0
    return [r1, g1, b1]


def hex_to_bit_color(color):
    hex_color = color.replace('#', '')
    r = int(hex_color.substring(0, 2), 16)
    g = int(hex_color.substring(2, 4), 16)
    b = int(hex_color.substring(4, 6), 16)
    r1 = 1 if r > 128 else 0
    g1 = 1 if g > 128 else 0
    b1 = 1 if b > 128 else 0
    return [r1, g1, b1]
