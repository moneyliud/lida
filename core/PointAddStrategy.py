from .Point3DToLida import PointAddInterface
import numpy as np


class CirclePointStrategy(PointAddInterface):
    def __init__(self):
        super().__init__()
        self.point_size = 1200
        self.point_num = 40

    def get_contour(self, x, y, color):
        theta = np.arange(0, 2 * np.pi, 2 * np.pi / self.point_num)
        radius = self.point_size / 2
        px = x + radius * np.cos(theta)
        py = y + radius * np.sin(theta)
        contour = []
        for j in range(len(px)):
            contour.append([int(px[j]), int(py[j]), color])
            # print(px[j], py[j])
        for i in range(5):
            contour.append(contour[0])
        return contour
        pass


class SinglePointStrategy(PointAddInterface):
    def __init__(self):
        super().__init__()

    def get_contour(self, x, y, color):
        contour = []
        contour.append([int(x), int(y), color])
        return contour
