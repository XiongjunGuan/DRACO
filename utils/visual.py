"""
Description:
Author: Xiongjun Guan
Date: 2024-06-13 16:15:18
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2025-02-27 19:39:49

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import gc

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as sndi


def draw_pose(ax, pose, length=100, color="blue"):
    x, y, theta = pose
    start = (x, y)
    end = (
        x - length * np.sin(theta * np.pi / 180.0),
        y - length * np.cos(theta * np.pi / 180.0),
    )
    ax.plot(start[0], start[1], marker="o", color=color)
    ax.arrow(
        start[0],
        start[1],
        end[0] - start[0],
        end[1] - start[1],
        width=2,
        fc=color,
        ec=color,
    )


def draw_img_with_pose(
    img,
    pose,
    save_path,
    scale=1,
    cmap="gray",
    vmin=None,
    vmax=None,
    mask=None,
    length=100,
    color="blue",
):
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if scale != 1:
        img = sndi.zoom(img, scale, order=1, cval=img[0].max())
        pose[:2] = pose[:2] * scale

    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    draw_pose(ax, pose, length=length, color=color)

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def draw_img_with_pose_2(
    img,
    pose,
    pose2,
    save_path,
    scale=1,
    cmap="gray",
    vmin=None,
    vmax=None,
    mask=None,
    length=100,
    color="red",
    color2="blue",
):
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if scale != 1:
        img = sndi.zoom(img, scale, order=1, cval=img[0].max())
        pose[:2] = pose[:2] * scale
        pose2[:2] = pose2[:2] * scale

    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    draw_pose(ax, pose, length=length, color=color)
    draw_pose(ax, pose2, length=length, color=color2)

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
