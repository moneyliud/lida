# -*- coding: utf-8 -*-

import serial
from threading import Thread


def serial_read(ser, a2):
    print("asdasd\n")
    while True:
        num = ser.inWaiting()
        if num > 0:
            msg = ser.read(num)
            print(msg)
    pass


if __name__ == '__main__':
    lida_file = "D:\\pyWorkspace\\lida\\test.ild"
    file = open(lida_file, 'rb+')
    data = file.read()
    file.close()
    file_len = len(data)
    print("file_len:", file_len)
    ser = serial.Serial("COM3", 115200)
    buffer_size = 2048
    t1 = Thread(target=serial_read, args=(ser, buffer_size))
    if ser.isOpen():
        t1.start()
        print("打开串口成功。")
        print(ser.name)
        ser.write(bytes([0xAA, 0xBB]))
        ser.write(file_len.to_bytes(4, "little"))
        write_len = 0
        while write_len < file_len:
            end = (write_len + buffer_size) if write_len + buffer_size < file_len else file_len
            ser.write(data[write_len:end])
            print(end)
            write_len += buffer_size
        ser.flush()
        t1.join()
    else:
        print("打开串口失败。")
