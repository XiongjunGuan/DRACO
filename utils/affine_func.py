"""
Description:
Author: Xiongjun Guan
Date: 2024-06-13 10:31:54
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-07-09 15:23:52

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as sndi


def affine_img(img, dx, dy, theta, pad_width=0, fit_value=255):
    """translation -> rotation

    Args:
        img (_type_): _description_
        dx (_type_): col pixel
        dy (_type_): row pixel
        theta (_type_): degree
        pad_width (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    if pad_width > 0:
        img = np.pad(
            img,
            [[pad_width, pad_width], [pad_width, pad_width]],
            "constant",
            constant_values=fit_value,
        )

    h, w = img.shape[:2]

    # translation
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img = cv2.warpAffine(
        img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=fit_value
    )

    # rotation
    center = (h // 2, w // 2)
    M = cv2.getRotationMatrix2D(center, theta, 1.0)
    img = cv2.warpAffine(
        img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=fit_value
    )
    if pad_width > 0:
        img = img[pad_width:-pad_width, pad_width:-pad_width]
    return img


def translate_vec(vec0, xc, yc, dx, dy, theta):
    """

    Args:
        vec0 (_type_): [x0,y0,theta0]
        xc (_type_): _description_
        yc (_type_): _description_
        dx (_type_): _description_
        dy (_type_): _description_
        theta (_type_): _description_

    Returns:
        _type_: _description_
    """
    vec0_x, vec0_y, vec0_theta = vec0

    x_mid = (vec0_x + dx) - xc
    y_mid = yc - (vec0_y + dy)

    phi = -np.deg2rad(theta)
    x_mid2 = x_mid * np.cos(phi) + y_mid * np.sin(phi)
    y_mid2 = y_mid * np.cos(phi) - x_mid * np.sin(phi)

    vec_x = xc + x_mid2
    vec_y = yc - y_mid2
    vec_theta = theta + vec0_theta

    return vec_x, vec_y, vec_theta


def rectify_pose(img, vec, pad_width=100):
    vec_x, vec_y, vec_theta = vec

    h, w = img.shape[:2]
    xc = w // 2
    yc = h // 2

    dx = xc - vec_x
    dy = yc - vec_y
    theta = -vec_theta

    img = affine_img(img, dx, dy, theta, pad_width=pad_width)

    return img


def fp2cap(img, scale=8.0 / 500):
    h, w = img.shape
    assert h == 512
    assert w == 512

    img = np.pad(img, ((100, 100), (100, 100)), mode="constant", constant_values=255)
    win_size = np.rint(1 / scale)
    img = sndi.uniform_filter(img, win_size, mode="constant")
    img = sndi.zoom(img, scale, order=1, cval=0)
    img = (255 - img)[2:-2, 2:-2]
    img = 255 * ((img - img.min()) / (img.max() - img.min()))
    return img
