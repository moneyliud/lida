import math

import numpy as np
from copy import deepcopy
from sklearn.cluster import dbscan
import cv2


def get_label_org_color(label, color_list):
    label_org_color = [None, None]
    label_index = [0, 1]
    for i in range(len(label)):
        if label_org_color[label[i][0]] is None:
            label_org_color[label[i][0]] = deepcopy(color_list[i])
        if label_org_color[0] is not None and label_org_color[1] is not None:
            break
        pass
    std1 = np.std(label_org_color[0])
    std2 = np.std(label_org_color[1])
    if std1 > std2:
        tmp = label_org_color[0]
        label_org_color[0] = label_org_color[1]
        label_org_color[1] = tmp
        label_index = [1, 0]
    return label_org_color, label_index


def match_label_color(label_org_color, cur_label, cur_color):
    cur_label_color_index = [None, None]
    for i in range(len(cur_label)):
        if cur_label_color_index[cur_label[i][0]] is None:
            a1 = np.abs(np.int32(cur_color[i]) - np.int32(label_org_color[0]))
            a2 = np.abs(np.int32(cur_color[i]) - np.int32(label_org_color[1]))
            sum1, sum2 = 0, 0
            for x in a1:
                sum1 += x
            for x in a2:
                sum2 += x
            if sum1 < sum2:
                cur_label_color_index[cur_label[i][0]] = 0
            else:
                cur_label_color_index[cur_label[i][0]] = 1
        if cur_label_color_index[0] is not None and cur_label_color_index[1] is not None:
            break
        pass
    return cur_label_color_index


def dbscan_circle(points, eps=3, min_samples=8):
    core_samples, cluster_ids = dbscan(points, eps=eps, min_samples=min_samples)
    sample_group = {}
    result = []
    for i in range(len(cluster_ids)):
        if cluster_ids[i] != -1:
            if not sample_group.__contains__(cluster_ids[i]):
                sample_group[cluster_ids[i]] = []
            sample_group[cluster_ids[i]].append(points[i])
    for i in sample_group:
        (x, y), radius = cv2.minEnclosingCircle(np.float32(sample_group[i]))
        result.append([int(x), int(y), int(radius * 1.5)])
    return result


# 根据hsv图像查找图像主体,计算Hsv主体点位凸包
def find_image_primary_area(image, tolerance=35, detect_type="gray"):
    # primary_color_h, mask, h_img = calculate_hsv_primary_area(image, hsv_tolerance=hsv_tolerance)
    mask = None
    if detect_type == "hsv":
        primary_color_h, mask, h_img = calculate_hsv_primary_area(image, hsv_tolerance=tolerance)
    else:
        mask = calculate_gray_primary_area(image, gray_tolerance=tolerance)
    mask = erode_dilate(mask, (3, 2), (16, 16))
    img1_contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    max_n_contour = find_max_n_contours(img1_contours, 1)
    image_range = cv2.drawContours(np.zeros((image.shape[0], image.shape[1])), [max_n_contour[0]], -1, (255, 255, 255),
                                   cv2.FILLED)
    return image_range, mask


def find_max_n_contours(contours, n, threshold=100):
    max_n_value = [0] * n
    max_n_contour = [None] * n
    # 查找前n大边缘作为轮廓
    for contour in contours:
        cur_area = len(contour)
        i = 0
        while i < n and cur_area <= max_n_value[i]:
            i += 1
        if i != n:
            for j in range(n - 1, i, -1):
                max_n_value[j] = max_n_value[j - 1]
                max_n_contour[j] = max_n_contour[j - 1]
            max_n_value[i] = cur_area
            max_n_contour[i] = contour
    max_n_contour_ret = []
    for i in range(len(max_n_value)):
        if max_n_value[i] > threshold or i < 5:
            max_n_contour_ret.append(max_n_contour[i])
    return max_n_contour_ret


def calculate_gray_primary_area(image, gray_tolerance=35):
    # img1_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # ret, img1_gray1 = cv2.threshold(img1_gray, gray_tolerance, 255, cv2.THRESH_BINARY)
    # # ret, img1_gray2 = cv2.threshold(img1_gray, 150, 255, cv2.THRESH_BINARY_INV)
    ret = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # h_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         # 亮度大于46 且灰度大于43 不为黑色也不为白色
    #         if hsv[i][j][2] > 46 and hsv[i][j][1] > 43:
    #             ret[i][j] = 255
    # 亮度大于46 且灰度大于43 不为黑色也不为白色
    ret = np.uint8(np.where((hsv[:, :, 2] > 60) & (hsv[:, :, 1] > 43), 255, 0))
    # # h 色相通道在335-25之间的红色，且s通道饱和度在70以上的判断为红色笔记
    # if img1_gray[i][j] != 0 and ((335 < hsv[i][j][0] <= 360) or (0 <= hsv[i][j][0] < 25)) and hsv[i][j][1] > 70:
    #     h_img[i][j] = 255
    # else:
    #     h_img[i][j] = 0
    return ret


# 根据hsv图像查找图像主体
def calculate_hsv_primary_area(image, hsv_tolerance=15):
    hist_threshold = 500
    img1_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 灰度图中亮度小的暗部直接赋为255
    ret, img1_gray = cv2.threshold(img1_gray, 35, 220, cv2.THRESH_BINARY)
    # 获取hsv图像 取h通道计算图像主体色相
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if img1_gray[i][j] != 0 and img1_gray[i][j] != 255:
                # 白色不参与主体颜色计算
                if is_white_by_hsv(hsv[i][j]):
                    h_img[i][j] = 0
                else:
                    h_img[i][j] = hsv[i][j][0]
            else:
                h_img[i][j] = 0
    # 计算图像主体颜色
    hist = cv2.calcHist([h_img], [0], None, [360], [0, 255])
    combine_num = 6
    hist_combine = []
    for i in range(int(len(hist) / combine_num)):
        sum_hist = 0.0
        for j in range(combine_num):
            sum_hist += hist[i * combine_num + j]
        hist_combine.append(sum_hist)
    # 黑色色阶不参与计算，直接赋值为0
    hist_combine[0] = np.array([0.0])
    max_hist_i_org = np.where(hist_combine == np.max(hist_combine))[0][0]
    max_hist_i = max_hist_i_org * combine_num
    # 图像主体颜色容差
    hsv_range_lower = max_hist_i - hsv_tolerance
    hsv_range_upper = max_hist_i
    for i in range(max_hist_i_org, len(hist_combine)):
        if hist_combine[i] < hist_threshold:
            hsv_range_upper += combine_num * (i - max_hist_i_org)
            break

    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if hsv_range_lower <= h_img[i][j] <= hsv_range_upper:
                mask[i][j] = 255
    return max_hist_i, mask, h_img


def is_white_by_hsv(hsv):
    if hsv[1] <= 30 and hsv[2] >= 221:
        return True
    return False


# 图像侵蚀+膨胀
def erode_dilate(img, kernel_erode, kernel_dilate):
    img_res = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_erode),
                        borderType=cv2.BORDER_CONSTANT,
                        borderValue=0)
    img_res = cv2.dilate(img_res, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_dilate),
                         borderType=cv2.BORDER_CONSTANT,
                         borderValue=0)
    return img_res


def sift_key_point_filter(key_point, dst, mask):
    ret_point, ret_dst = [], []
    for i in range(len(key_point)):
        if mask[int(key_point[i].pt[1]), int(key_point[i].pt[0])] == 255:
            ret_point.append(key_point[i])
            ret_dst.append(dst[i])
    return np.array(ret_point), np.array(ret_dst)


def calculate_grad_direction(image1_gary):
    x_grad = cv2.Sobel(image1_gary, cv2.CV_64F, 1, 0, ksize=5)
    y_grad = cv2.Sobel(image1_gary, cv2.CV_64F, 0, 1, ksize=5)
    for i in range(x_grad.shape[0]):
        for j in range(x_grad.shape[1]):
            dir_len = math.sqrt(x_grad[i, j] ** 2 + y_grad[i, j] ** 2)
            if dir_len != 0:
                x_grad[i, j] = x_grad[i, j] / dir_len
                y_grad[i, j] = y_grad[i, j] / dir_len
    return np.concatenate((np.expand_dims(x_grad, 2), np.expand_dims(y_grad, 2)), 2)


def draw_direction_vector(contours, grad_dir):
    sub_contours_new = np.zeros((contours.shape[0], contours.shape[1]), np.uint8)
    for i in range(contours.shape[0]):
        for j in range(contours.shape[1]):
            if contours[i, j] == 255:
                for k in range(-20, 20):
                    y_index = i + int(k * grad_dir[i, j][1])
                    x_index = j + int(k * grad_dir[i, j][0])
                    if y_index < contours.shape[0] and x_index < contours.shape[1]:
                        sub_contours_new[y_index][x_index] = 255


def search_exist_point(vertex1, vertex2, vertex3, vertex4, contour_img1, contour_img2):
    vertex1 = limit_value(vertex1, 0, contour_img1.shape[1] - 1)
    vertex2 = limit_value(vertex2, 0, contour_img1.shape[1] - 1)
    vertex3 = limit_value(vertex3, 0, contour_img1.shape[0] - 1)
    vertex4 = limit_value(vertex4, 0, contour_img1.shape[0] - 1)
    exist_img1 = False
    exist_img2 = False
    for i in range(vertex1, vertex2):
        if contour_img1[vertex3][i] == 255 or contour_img1[vertex4][i] == 255:
            exist_img1 = True
        if contour_img2[vertex3][i] == 255 or contour_img2[vertex4][i] == 255:
            exist_img2 = True
    for i in range(vertex3, vertex4):
        if contour_img1[i][vertex1] == 255 or contour_img1[i][vertex2] == 255:
            exist_img1 = True
        if contour_img2[i][vertex1] == 255 or contour_img2[i][vertex2] == 255:
            exist_img2 = True
    return exist_img1, exist_img2


def limit_value(value, lower_bound, upper_bound):
    if value < lower_bound:
        value = lower_bound
    if value > upper_bound:
        value = upper_bound
    return value


def find_nearest_point(search_x, search_y, search_x_, search_y_, contour_img):
    find_flag = False
    if contour_img.shape[1] > search_x >= 0 and contour_img.shape[0] > search_y >= 0 and \
            contour_img[search_y, search_x] == 255:
        find_flag = True
    if contour_img.shape[1] > search_x_ >= 0 and 0 <= search_y_ < contour_img.shape[0] and \
            contour_img[search_y_, search_x_] == 255:
        find_flag = True
    if not (contour_img.shape[1] > search_x >= 0 and contour_img.shape[0] > search_y >= 0) and \
            not (contour_img.shape[1] > search_x_ >= 0 and 0 <= search_y_ < contour_img.shape[0]):
        find_flag = True
    return find_flag


def fit_image_brightness(image1, image2, mask):
    hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    for i in range(hsv1.shape[0]):
        for j in range(hsv1.shape[1]):
            if mask[i, j] != 255:
                hsv1[i, j] = [0, 0, 0]
                hsv2[i, j] = [0, 0, 0]
    v_img1 = hsv1[:, :, 2]
    v_img2 = hsv2[:, :, 2]
    v_mean1 = np.mean(v_img1)
    v_mean2 = np.mean(v_img2)
    alpha = v_mean1 / v_mean2
    for i in range(v_img2.shape[0]):
        for j in range(v_img2.shape[1]):
            tmp = v_img2[i, j] * alpha
            if tmp > 255:
                tmp = 255
            v_img2[i, j] = tmp
    hsv2[:, :, 2] = v_img2
    ret2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    return image1, ret2, hsv1, hsv2


def find_contour_rect(contour):
    min_x, min_y, max_x, max_y = 999999, 999999, 0, 0
    for i in range(len(contour)):
        min_x = min(min_x, contour[i][0][0])
        min_y = min(min_y, contour[i][0][1])
        max_x = max(max_x, contour[i][0][0])
        max_y = max(max_y, contour[i][0][1])
    return min_x, min_y, max_x, max_y


def get_larger_rect(min_x, min_y, max_x, max_y, detect_range, alpha=1.5):
    middle_x = (max_x + min_x) / 2
    middle_y = (max_y + min_y) / 2
    half_len_x = max((max_x - min_x) * alpha / 2, 20)
    half_len_y = max((max_y - min_y) * alpha / 2, 20)
    left = max(middle_x - half_len_x, 0)
    right = min(middle_x + half_len_x, detect_range.shape[1])
    top = max(middle_y - half_len_y, 0)
    down = min(middle_y + half_len_y, detect_range.shape[0])
    return int(left), int(right), int(top), int(down)


def get_image_by_mask(image, mask):
    ret = np.zeros(image.shape, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i, j] == 255:
                ret[i, j] = image[i, j]
    return ret


def mean_rgb_img(image):
    color_array = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if not (image[i, j] == (0, 0, 0)).all():
                color_array.append(image[i, j])
    mean = np.mean(color_array, 0)
    return mean


def set_nonzero_color(image, color):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if not (image[i, j] == (0, 0, 0)).all():
                image[i, j] = color


# bgr
def calculate_rgb_diff(bgr1, bgr2):
    bgr1 = np.float32(bgr1)
    bgr2 = np.float32(bgr2)
    r_mean = (bgr1[2] + bgr2[2]) / 2
    r = bgr1[2] - bgr2[2]
    g = bgr1[1] - bgr2[1]
    b = bgr1[0] - bgr2[0]
    ret = math.sqrt((2 + r_mean / 256) * (r ** 2) + 4 * (g ** 2) + (2 + (255 - r_mean) / 256) * (b ** 2))
    return ret


def dbscan_find_contours(image, eps=1.5, min_samples=2):
    points = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if not (image[i, j] == (0, 0, 0)).all():
                points.append([j, i])
    core_samples, cluster_ids = dbscan(points, eps=eps, min_samples=min_samples)
    sample_group = {}
    result = []
    for i in range(len(cluster_ids)):
        if cluster_ids[i] != -1:
            if cluster_ids[i] not in sample_group:
                sample_group[cluster_ids[i]] = []
            sample_group[cluster_ids[i]].append([points[i]])
    for i in sample_group:
        result.append(np.array(sample_group[i]))
    return result


def set_zero_by_mask(image1, mask):
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if (mask[i, j] == 0).all():
                image1[i, j] = (0, 0, 0)


def is_point_in_rect(point, min_x, min_y, max_x, max_y):
    return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y


def remove_duplicated_contours(contour):
    ret = []
    for i in range(len(contour)):
        point_map = {}
        point_list = []
        for j in range(len(contour[i])):
            point_key = str(contour[i][j][0][0]) + "-" + str(contour[i][j][0][1])
            if point_key not in point_map:
                point_list.append(contour[i][j])
                point_map[point_key] = True
        ret.append(np.array(point_list))
    return ret
