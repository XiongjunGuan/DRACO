"""
Description:
Author: Xiongjun Guan
Date: 2023-12-11 10:21:53
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2025-02-27 19:41:17

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import argparse
import copy
import os
import os.path as osp
import random
import time
from glob import glob

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from data_loader_test import get_dataloader_test
from models.DRACO_Single import DRACO_Single
from utils.trans_est import classify2vector_rot, classify2vector_trans
from utils.visual import draw_img_with_pose


def set_seed(seed=7):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    set_seed(7)

    parser = argparse.ArgumentParser(description="Generate parameters")
    parser.add_argument(
        "-dataset",
        type=str,
        default="./examples/fp-inp",
    )
    parser.add_argument(
        "-model_dir",
        type=str,
        default="./ckpts/DRACO_Single-fp-180",
    )
    parser.add_argument(
        "-save_dir",
        type=str,
        default="./examples/fp-res",
    )
    parser.add_argument(
        "-batch_size",
        type=int,
        default=4,
        help="batch size",
    )
    parser.add_argument(
        "-cuda_ids",
        dest="cuda_ids",
        default="0",
    )
    args = parser.parse_args()

    cuda_ids = args.cuda_ids
    if "," in cuda_ids:
        cuda_ids = [int(x) for x in args.cuda_ids.split(",")]
    else:
        cuda_ids = [int(cuda_ids)]
    batch_size = args.batch_size

    # --- load model
    model_dir = args.model_dir
    pth_path = osp.join(model_dir, "best.pth")
    config_path = osp.join(model_dir, "config.yaml")
    with open(config_path, "r") as config:
        cfg = yaml.safe_load(config)

    device = torch.device(
        "cuda:{}".format(str(cuda_ids[0])) if torch.cuda.is_available() else "cpu"
    )

    model = DRACO_Single(
        inp_mode=cfg["model_cfg"]["inp_mode"],
        trans_out_form=cfg["model_cfg"]["trans_out_form"],
        trans_num_classes=cfg["model_cfg"]["trans_num_classes"],
        rot_out_form=cfg["model_cfg"]["rot_out_form"],
        rot_num_classes=cfg["model_cfg"]["rot_num_classes"],
    )
    model.load_state_dict(torch.load(pth_path, map_location=f"cuda:{cuda_ids[0]}"))

    model = torch.nn.DataParallel(
        model,
        device_ids=cuda_ids,
        output_device=cuda_ids[0],
    ).to(device)
    model.eval()

    # --- load test info
    test_dir = args.dataset
    test_fp_img_dir = osp.join(test_dir)

    ftitle_lst = glob(osp.join(test_dir, "*.png"))
    ftitle_lst = [osp.basename(x).replace(".png", "") for x in ftitle_lst]
    img_lst = []
    for ftitle in ftitle_lst:
        img_lst.append(osp.join(test_dir, ftitle + ".png"))

    # --- set save dir
    save_dir = args.save_dir
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    valid_loader = get_dataloader_test(
        fp_lst=img_lst,
        patch_lst=None,
        mask_lst=None,
        cap_lst=None,
        inp_mode=cfg["model_cfg"]["inp_mode"],
        batch_size=batch_size,
        shuffle=False,
    )

    # --- pred config
    trans_out_form = cfg["model_cfg"]["trans_out_form"]
    trans_num_classes = cfg["model_cfg"]["trans_num_classes"]
    rot_out_form = cfg["model_cfg"]["rot_out_form"]
    rot_num_classes = cfg["model_cfg"]["rot_num_classes"]

    # --- test phase
    with torch.no_grad():
        pbar = tqdm(valid_loader)
        for fp, patch, cap, mask, fp_path in pbar:

            fp = fp.float().to(device)
            [pred_xy, pred_theta] = model(fp)

            vec_xy = classify2vector_trans(
                pred_xy, out_form=trans_out_form, trans_num_classes=trans_num_classes
            )
            vec_theta = classify2vector_rot(
                pred_theta, out_form=rot_out_form, rot_num_classes=rot_num_classes
            )

            imgs = fp.cpu().numpy()
            vec_xys = vec_xy.cpu().numpy()
            vec_thetas = vec_theta.cpu().numpy()

            img_size = fp.shape[-1]
            for i in range(imgs.shape[0]):
                # img_i = 255 - imgs[i, 0, :, :] * 255
                vec_xy_i = vec_xys[i, :] + img_size // 2
                vec_theta_i = vec_thetas[i, -1]
                vec_pred = [vec_xy_i[0], vec_xy_i[1], vec_theta_i]

                fpath_i = fp_path[i]
                ftitle_i = osp.basename(fpath_i).replace(".png", "")
                img_path_i = osp.join(test_fp_img_dir, ftitle_i + ".png")
                img_i = cv2.imread(img_path_i, 0)

                save_path_i = osp.join(save_dir, ftitle_i + ".png")
                if osp.exists(save_path_i):
                    os.remove(save_path_i)

                draw_img_with_pose(img_i, vec_pred, save_path=save_path_i)

                save_path_i = osp.join(save_dir, ftitle_i + ".txt")
                np.savetxt(save_path_i, vec_pred, fmt="%.2f")

                pass

        pbar.close()
