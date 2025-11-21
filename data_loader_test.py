"""
Description: build [train / valid / test] dataloader
Author: Xiongjun Guan
Date: 2023-03-01 19:41:05
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-03-20 21:01:47

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import copy
import logging
import os.path as osp
import random

import cv2
import numpy as np
from scipy import signal
from torch.utils.data import DataLoader, Dataset
from utils.affine_func import affine_img, fp2cap, translate_vec
from utils.visual import draw_img_with_pose


def prob_normalization(prob, eps=1e-6):
    prob = prob / np.clip(np.sum(prob, axis=0, keepdims=True), eps, np.inf)
    return prob


# class load_dataset_train(Dataset):

#     def __init__(
#         self,
#         fp_lst: None,
#         patch_lst: None,
#         mask_lst: None,
#         cap_lst: None,
#         pose_lst: None,
#         inp_mode='fp',
#         inp_patch_size=None,
#         trans_num_classes=120,
#         rot_num_classes=120,
#         apply_aug=False,
#         trans_aug=40,
#         rot_aug=180,
#     ):
#         self.fp_lst = fp_lst
#         self.patch_lst = patch_lst
#         self.mask_lst = mask_lst
#         self.cap_lst = cap_lst
#         self.pose_lst = pose_lst
#         self.inp_mode = inp_mode
#         self.inp_patch_size = inp_patch_size
#         self.trans_num_classes = trans_num_classes
#         self.rot_num_classes = rot_num_classes
#         self.apply_aug = apply_aug
#         self.trans_aug = trans_aug
#         self.rot_aug = rot_aug

#     def __len__(self):
#         return len(self.pose_lst)

#     def __getitem__(self, idx):

#         fp_path = self.fp_lst[idx]
#         fp = cv2.imread(fp_path, 0).astype(np.float32)

#         if 'mask' in self.inp_mode:
#             mask_path = self.mask_lst[idx]
#             mask = cv2.imread(mask_path, 0).astype(np.float32)
#         else:
#             mask = np.array([-1])

#         if ('cap' in self.inp_mode) and (self.cap_lst is not None):
#             cap_path = self.cap_lst[idx]
#             cap = cv2.imread(cap_path, 0).astype(np.float32)
#         else:
#             cap = np.array([-1])

#         if ('patch' in self.inp_mode) and (self.patch_lst is not None):
#             patch_path = self.patch_lst[idx]
#             patch = cv2.imread(patch_path, 0).astype(np.float32)
#         else:
#             patch = np.array([-1])

#         pose_path = self.pose_lst[idx]

#         pose_data = np.loadtxt(pose_path, dtype=str)
#         vec0 = np.array(
#             list(map(float, [x.replace(',', '') for x in pose_data])))

#         if self.apply_aug:
#             dx = random.randint(-self.trans_aug, self.trans_aug)
#             dy = random.randint(-self.trans_aug, self.trans_aug)
#             dtheta = random.randint(-self.rot_aug, self.rot_aug)

#             h, w = fp.shape[:2]
#             xc = w // 2
#             yc = h // 2
#             vec = translate_vec(vec0, xc, yc, dx, dy, dtheta)
#             if ('fp' in self.inp_mode) or (('cap' in self.inp_mode) and
#                                            (self.cap_lst is None)) or (
#                                                ('patch' in self.inp_mode) and
#                                                (self.patch_lst is None)):
#                 fp = affine_img(fp,
#                                 dx,
#                                 dy,
#                                 dtheta,
#                                 pad_width=100,
#                                 fit_value=255)
#             if 'mask' in self.inp_mode:
#                 mask = affine_img(mask,
#                                   dx,
#                                   dy,
#                                   dtheta,
#                                   pad_width=100,
#                                   fit_value=0)
#         else:
#             h, w = fp.shape[:2]
#             xc = w // 2
#             yc = h // 2
#             vec = translate_vec(vec0, xc, yc, 0, 0, 0)

#         if ('patch' in self.inp_mode) and (self.patch_lst is None):
#             patch_mask = np.zeros_like(fp)
#             hc, wc = h // 2, w // 2
#             hps = self.inp_patch_size // 2
#             patch = fp[hc - hps:hc + hps, wc - hps:wc + hps]
#         if ('cap' in self.inp_mode) and (self.cap_lst is None):
#             cap = fp2cap(fp)

#         if 'patchPad' in self.inp_mode:
#             new_patch = 255 * np.ones_like(fp[:, :])
#             h, w = new_patch.shape
#             hc, wc = h // 2, w // 2
#             hps = self.inp_patch_size // 2
#             new_patch[hc - hps:hc + hps, wc - hps:wc + hps] = patch
#             patch = copy.deepcopy(new_patch)

#         # draw_img_with_pose(
#         #     np.squeeze(fp).astype(np.int32), vec,
#         #     "/home/guanxiongjun/code/FingerPose/tmp/0.png")

#         fp = (255.0 - fp) / 255.0  # 0 to background
#         fp = fp[None, :, :]
#         if 'patch' in self.inp_mode:
#             patch = (255.0 - patch) / 255.0  # 0 to background
#             patch = patch[None, :, :]
#         if 'cap' in self.inp_mode:
#             cap = cap / 255.0
#             cap = cap[None, :, :]
#         if 'mask' in self.inp_mode:
#             mask = mask / 255.0
#             mask = mask[None, :, :]

#         vec_x, vec_y, vec_theta = vec

#         _, h, w = fp.shape
#         vec_x -= (w // 2)
#         vec_y -= (h // 2)

#         vec_cos = np.cos(np.deg2rad(vec_theta))
#         vec_sin = np.sin(np.deg2rad(vec_theta))
#         vec_theta = np.rad2deg(np.arctan2(vec_sin, vec_cos))  # [-180,180]
#         target = np.array([vec_x, vec_y, vec_cos, vec_sin, vec_theta])

#         gaussian_pdf = signal.gaussian(361, 2.5)
#         rot_arr = np.linspace(-180, 180, self.rot_num_classes)
#         delta = np.array(np.abs(vec_theta - rot_arr), dtype=int)
#         delta = np.minimum(delta, 360 - delta) + 180
#         target_prob_theta = gaussian_pdf[delta]

#         trans_const = 256
#         gaussian_pdf = signal.gaussian(trans_const * 2 + 1, 3.5)
#         trans_arr = np.linspace(-trans_const, trans_const,
#                                 self.trans_num_classes // 2)
#         delta = np.array(np.abs(vec_x - trans_arr), dtype=int)
#         delta[delta > trans_const] = trans_const
#         delta = delta + trans_const
#         target_prob_x = gaussian_pdf[delta]
#         delta = np.array(np.abs(vec_y - trans_arr), dtype=int)
#         delta[delta > trans_const] = trans_const
#         delta = delta + trans_const
#         target_prob_y = gaussian_pdf[delta]

#         target_prob_theta = prob_normalization(target_prob_theta)
#         target_prob_x = prob_normalization(target_prob_x)
#         target_prob_y = prob_normalization(target_prob_y)

#         return fp, patch, cap, mask, target, target_prob_x, target_prob_y, target_prob_theta, fp_path


class load_dataset_test(Dataset):

    def __init__(
        self,
        fp_lst: None,
        patch_lst: None,
        mask_lst: None,
        cap_lst: None,
        inp_mode="fp",
    ):
        self.fp_lst = fp_lst
        self.patch_lst = patch_lst
        self.mask_lst = mask_lst
        self.cap_lst = cap_lst
        self.inp_mode = inp_mode

    def __len__(self):
        if self.inp_mode == "fp":
            return len(self.fp_lst)
        elif self.inp_mode == "cap":
            return len(self.cap_lst)
        elif self.inp_mode == "patch":
            return len(self.patch_lst)
        elif self.inp_mode == "patch_cap":
            return len(self.patch_lst)

    def __getitem__(self, idx):
        if self.inp_mode == "fp":
            fp_path = self.fp_lst[idx]
            fp = cv2.imread(fp_path, 0).astype(np.float32)
            load_path = fp_path
        else:
            fp = np.array([-1])

        if "mask" in self.inp_mode:
            mask_path = self.mask_lst[idx]
            mask = cv2.imread(mask_path, 0).astype(np.float32)
            load_path = mask_path
        else:
            mask = np.array([-1])

        if "cap" in self.inp_mode:
            cap_path = self.cap_lst[idx]
            cap = cv2.imread(cap_path, 0).astype(np.float32)
            load_path = cap_path
        else:
            cap = np.array([-1])

        if "patch" in self.inp_mode:
            patch_path = self.patch_lst[idx]
            patch = cv2.imread(patch_path, 0).astype(np.float32)
            load_path = patch_path
        else:
            patch = np.array([-1])

        if self.inp_mode == "fp":
            fp = (255.0 - fp) / 255.0  # 0 to background
            fp = fp[None, :, :]
        if "patch" in self.inp_mode:
            patch = (255.0 - patch) / 255.0  # 0 to background
            patch = patch[None, :, :]
        if "cap" in self.inp_mode:
            cap = cap / 255.0
            cap = cap[None, :, :]
        if "mask" in self.inp_mode:
            mask = mask / 255.0
            mask = mask[None, :, :]

        return fp, patch, cap, mask, load_path


# def get_dataloader_valid_test(
#     fp_lst: None,
#     patch_lst: None,
#     mask_lst: None,
#     cap_lst: None,
#     pose_lst: None,
#     inp_mode='fp',
#     inp_patch_size=None,
#     trans_num_classes=120,
#     rot_num_classes=120,
#     apply_aug=False,
#     trans_aug=40,
#     rot_aug=180,
#     batch_size=1,
#     shuffle=False,
# ):
#     # Create dataset
#     try:
#         dataset = load_dataset_train(
#             fp_lst=fp_lst,
#             patch_lst=patch_lst,
#             mask_lst=mask_lst,
#             cap_lst=cap_lst,
#             pose_lst=pose_lst,
#             inp_mode=inp_mode,
#             inp_patch_size=inp_patch_size,
#             trans_num_classes=trans_num_classes,
#             rot_num_classes=rot_num_classes,
#             apply_aug=apply_aug,
#             trans_aug=trans_aug,
#             rot_aug=rot_aug,
#         )
#     except Exception as e:
#         logging.error("Error in DataLoader: ", repr(e))
#         return

#     valid_loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=4,
#         pin_memory=True,
#     )
#     logging.info(f"n_valid:{len(dataset)}")

#     return valid_loader


def get_dataloader_test(
    fp_lst: None,
    patch_lst: None,
    mask_lst: None,
    cap_lst: None,
    inp_mode="fp",
    batch_size=1,
    shuffle=False,
):
    # Create dataset
    try:
        dataset = load_dataset_test(
            fp_lst=fp_lst,
            patch_lst=patch_lst,
            mask_lst=mask_lst,
            cap_lst=cap_lst,
            inp_mode=inp_mode,
        )
    except Exception as e:
        logging.error("Error in DataLoader: ", repr(e))
        return

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )
    logging.info(f"n_test:{len(dataset)}")

    return test_loader
