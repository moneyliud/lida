from typing import List


class LidaPoints:
    def __init__(self):
        self.BYTE_ORDER = "big"
        self.x = 0
        self.y = 0
        self.z = 0
        self.enable = 0
        self.color = 0
        pass

    def to_bytes(self):
        point_bytes = bytearray(8)
        point_bytes[0:2] = self.x.to_bytes(2, self.BYTE_ORDER, signed=True)
        point_bytes[2:4] = self.y.to_bytes(2, self.BYTE_ORDER, signed=True)
        point_bytes[4:6] = self.z.to_bytes(2, self.BYTE_ORDER, signed=True)
        point_bytes[6] = self.enable
        point_bytes[7] = self.color
        return point_bytes


class LidaHeader:
    def __init__(self):
        self.BYTE_ORDER = "big"
        self.ENCODING = "utf-8"
        self.LIDA = b"ILDA"  # 4 byte
        self.type = 0  # 4byte
        self.name = ""  # 8byte
        self.company = ""  # 8byte
        self.frame_length = 0  # 2byte
        self.frame_index = 0  # 2byte
        self.total_frame = 0  # 2byte
        self.duration = 0  # 持续时间
        self.head = 0  # 1byte
        self.end = 0  # 1byte

    def to_bytes(self):
        header_bytes = bytearray(32)
        header_bytes[0:4] = self.LIDA
        header_bytes[4:8] = self.type.to_bytes(4, self.BYTE_ORDER)
        header_bytes[8:16] = bytes(self.name.ljust(8)[0:8], self.ENCODING)
        header_bytes[15] = 0x20
        header_bytes[16:24] = bytes(self.company.ljust(8)[0:8], self.ENCODING)
        header_bytes[23] = 0x20
        header_bytes[24:26] = self.frame_length.to_bytes(2, self.BYTE_ORDER)
        header_bytes[26:28] = self.frame_index.to_bytes(2, self.BYTE_ORDER)
        header_bytes[28:30] = self.total_frame.to_bytes(2, self.BYTE_ORDER)
        header_bytes[30] = 0x00
        header_bytes[31] = self.duration.to_bytes(1, self.BYTE_ORDER)[0]
        return header_bytes


class LidaFrame:
    def __init__(self):
        self.header: LidaHeader = LidaHeader()
        self.points: List[LidaPoints] = []

    def to_bytes(self):
        frame_bytes = bytearray()
        frame_bytes += self.header.to_bytes()
        for point in self.points:
            frame_bytes += point.to_bytes()
        return frame_bytes


class LidaFile:

    def __init__(self):
        self.frames: List[LidaFrame] = []
        self.cur_frame_index = None
        self.name = ""
        self.company = ""
        self.end = bytes(
            [0x49, 0x4C, 0x44, 0x41, 0x00, 0x00, 0x00, 0x00,
             0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
             0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x3D, 0x00, 0x00])
        pass

    def to_bytes(self):
        self.refresh_frame_header()
        file_bytes = bytearray()
        i = 0
        for frame in self.frames:
            frame.header.name = str(i).zfill(3) + self.name
            frame.header.company = self.company
            file_bytes += frame.to_bytes()
            i += 1
        file_bytes += self.end
        return file_bytes

    def add_point(self, x, y, z, color, enable: bool = True):
        point = LidaPoints()
        point.x = x
        point.y = y
        point.z = z
        if enable:
            point.enable = 0
            point.color = color
            pass
        else:
            # 0x40 disable
            point.enable = 0x40
            point.color = 0
        self.frames[self.cur_frame_index].points.append(point)
        pass

    def new_frame(self):
        frame = LidaFrame()
        self.frames.append(frame)
        if self.cur_frame_index is None:
            self.cur_frame_index = 0
        else:
            self.cur_frame_index += 1
        pass

    def refresh_frame_header(self):
        total_frames = len(self.frames)
        for i in range(total_frames):
            frame = self.frames[i]
            frame.header.frame_index = i
            frame.header.frame_length = len(frame.points)
            frame.header.total_frame = total_frames
