import cv2
from core.lidaGenerator import LidaFile
from core.imageConverter import ImageILDAConverter

if __name__ == '__main__':
    lidaFile = LidaFile()
    lidaFile.name = "test123"
    lidaFile.company = "company0"
    image = cv2.imread("C:\\Users\\Administrator\\Pictures\\test18.jpg")
    converter = ImageILDAConverter()
    converter.save_image = True
    points = converter.convert(image)
    lidaFile.new_frame()
    print(len(points))
    for i in range(len(points)):
        point = points[i]
        lidaFile.add_point(point[0], point[1], point[2], point[3], point[4])
    file = open("./test.ild", 'wb')
    file.write(lidaFile.to_bytes())
    # cv2.waitKey(0)
