# -*- coding: utf-8 -*-

import cv2
import numpy as np


def read_YUV420(image, rows, cols):
    # create Y
    gray = np.zeros((rows, cols), np.uint8)
    # print(type(gray))
    # print(gray.shape)

    # create U,V
    img_U = np.zeros((int(rows / 2), int(cols / 2)), np.uint8)
    img_V = np.zeros((int(rows / 2), int(cols / 2)), np.uint8)
    # print(type(image))
    idx = 0
    for i in range(rows):
        for j in range(cols):
            gray[i, j] = image[idx]
            idx += 1

    for i in range(int(rows / 2)):
        for j in range(int(cols / 2)):
            img_U[i, j] = image[idx]
            idx += 1

    for i in range(int(rows / 2)):
        for j in range(int(cols / 2)):
            img_V[i, j] = image[idx]
            idx += 1

    return [gray, img_U, img_V]


def merge_YUV2RGB_v1(Y, U, V):
    enlarge_U = cv2.resize(U, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    enlarge_V = cv2.resize(V, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # 合并YUV3通道
    img_YUV = cv2.merge([Y, enlarge_U, enlarge_V])

    rgb = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
    return rgb


def merge_YUV2RGB_v2(Y, U, V):

    rows, cols = Y.shape[:2]

    shrink_Y = cv2.resize(Y, (cols / 2, rows / 2), interpolation=cv2.INTER_AREA)

    img_YUV = cv2.merge([shrink_Y, U, V])

    dst = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
    cv2.COLOR_YUV2BGR_I420
    
    enlarge_dst = cv2.resize(dst, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return enlarge_dst


if __name__ == '__main__':
    rows = 480
    cols = 640
    image_path = 'C:\\yuv\\jpgimage1_image_640_480.yuv'

    Y, U, V = read_YUV420(image_path, rows, cols)

    dst = merge_YUV2RGB_v1(Y, U, V)

    cv2.imshow("dst", dst)
    cv2.waitKey(0)
