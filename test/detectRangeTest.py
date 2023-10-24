import cv2
import core.detectUtil as detectUtil
import numpy as np


def erode_dilate(img, kernel_erode, kernel_dilate):
    img_res = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_erode),
                        borderType=cv2.BORDER_CONSTANT,
                        borderValue=0)
    img_res = cv2.dilate(img_res, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_dilate),
                         borderType=cv2.BORDER_CONSTANT,
                         borderValue=0)
    return img_res


if __name__ == "__main__":
    image = cv2.imread("../org_X20105-test5_U94tDoV.jpg")
    mask = detectUtil.calculate_gray_primary_area(image, gray_tolerance=35)
    # mask = erode_dilate(mask, (3, 2), (8, 8))
    cv2.imwrite("mask.jpg", mask)
    img1_contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    max_contour = detectUtil.find_max_n_contours(img1_contours, 1)[0]
    image_range = cv2.drawContours(np.zeros((image.shape[0], image.shape[1])), [max_contour], -1, (255, 255, 255),
                                   0)
    cv2.imshow("image_range.jpg", image_range)
    cv2.waitKey(0)
    # cv2.imwrite("image_range.jpg", image_range)
