from .Point3DToLida import PointAddInterface
import math


class CirclePointStrategy(PointAddInterface):
    def __init__(self):
        super().__init__()
        self.point_size = 4

    def get_contour(self, x, y, color):
        contour = []
        # ÉÏ°ëÔ²
        pre_j = 0
        reverse_dir = False
        for i in range(-int(self.point_size / 2), int(self.point_size / 2) + 1):
            len_range = int(self.point_size / 2)
            if pre_j >= len_range:
                reverse_dir = True
            if not reverse_dir:
                direct = 1
                start = pre_j
                end = len_range + 1
            else:
                direct = -1
                start = pre_j
                end = -1

            for j in range(start, end, direct):
                x1 = x + i
                y1 = y + j
                radius = math.pow((x1 - x) ** 2 + (y1 - y) ** 2, 0.5)
                if self.point_size / 2 <= radius < self.point_size / 2 + 1:
                    pre_j = j
                    contour.append([x1, y1, color])
                    break
        # ÏÂ°ëÔ²
        reverse_dir = False
        pre_j = 0
        for i in range(int(self.point_size / 2), -int(self.point_size / 2) - 1, -1):
            len_range = -int(self.point_size / 2)
            if pre_j <= len_range:
                reverse_dir = True
            if not reverse_dir:
                direct = -1
                start = pre_j
                end = len_range - 1
            else:
                direct = 1
                start = pre_j
                end = 0
            for j in range(start, end, direct):
                x1 = x + i
                y1 = y + j
                if self.point_size / 2 <= math.pow((x1 - x) ** 2 + (y1 - y) ** 2, 0.5) < self.point_size / 2 + 1:
                    pre_j = j
                    contour.append([x1, y1, color])
                    break
        if self.point_size > 50:
            contour_interval = int(self.point_size / 15)
            contour = contour[::contour_interval]
        return contour
        pass
