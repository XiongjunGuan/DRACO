"""
Description:
Author: Xiongjun Guan
Date: 2023-03-01 19:41:05
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-03-06 10:35:12

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import argparse
import datetime
import logging
import math
import os
import os.path as osp
import random
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm

from args import get_args
from data_loader import get_dataloader_train, get_dataloader_valid
from losses.loss import (
    DistillationLoss_CLIP,
    DistillationLoss_Feature,
    DistillationLoss_Relation,
    DistillationLoss_Response,
    EvalLoss,
    FinalLoss,
    RkdDistance,
)
from models.DRACO import DRACO
from models.DRACO_Single import DRACO_Single
from utils.trans_est import classify2vector_rot, classify2vector_trans


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_model(model, save_path):
    if hasattr(model, "module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(
        model_state,
        osp.join(save_path),
    )
    return


def train(
    model,
    teacher_model,
    train_dataloader,
    valid_dataloader,
    device,
    cfg,
    save_dir=None,
    save_checkpoint=15,
):
    # -------------- init settings-------------- #
    model_name = cfg["model_cfg"]["model_name"]
    lr = cfg["train_cfg"]["lr"]
    end_lr = cfg["train_cfg"]["end_lr"]
    optim = cfg["train_cfg"]["optimizer"]
    scheduler_type = cfg["train_cfg"]["scheduler_type"]
    num_epoch = cfg["train_cfg"]["epochs"]
    trans_out_form = cfg["model_cfg"]["trans_out_form"]
    trans_num_classes = cfg["model_cfg"]["trans_num_classes"]
    rot_out_form = cfg["model_cfg"]["rot_out_form"]
    rot_num_classes = cfg["model_cfg"]["rot_num_classes"]
    supervise_mode = cfg["loss_cfg"]["supervise_mode"]
    trans_loss_form = cfg["loss_cfg"]["trans_loss_form"]
    rot_loss_form = cfg["loss_cfg"]["rot_loss_form"]
    trans_loss_weight = cfg["loss_cfg"]["trans_loss_weight"]

    if valid_dataloader is None:
        valid = False
    else:
        valid = True

    # -------------- some global functions -------------- #
    criterion = FinalLoss(
        supervise_mode=supervise_mode,
        trans_loss_form=trans_loss_form,
        rot_out_form=rot_out_form,
        rot_loss_form=rot_loss_form,
        trans_loss_weight=trans_loss_weight,
    )
    criterion_eval = EvalLoss()

    distrill_criterions_dict = {}
    if "Relation" == args.distrill_mode:
        distrill_criterions_dict["distrill_Relation"] = DistillationLoss_Relation()
    if "Feature" == args.distrill_mode:
        distrill_criterions_dict["distrill_Feature"] = DistillationLoss_Feature()
    if "Response" == args.distrill_mode:
        distrill_criterions_dict["distrill_Response"] = DistillationLoss_Response()
    if "RkdDistance" == args.distrill_mode:
        distrill_criterions_dict["distrill_RkdDistance"] = RkdDistance()
    if "CLIP" == args.distrill_mode:
        distrill_criterions_dict["distrill_CLIP"] = DistillationLoss_CLIP()

    # -------------- select optimizer -------------- #
    if optim == "sgd":
        optimizer = torch.optim.SGD(
            (param for param in model.parameters() if param.requires_grad),
            lr=lr,
            weight_decay=0,
        )
    elif optim == "adam":
        optimizer = torch.optim.Adam(
            (param for param in model.parameters() if param.requires_grad),
            lr=lr,
            weight_decay=1e-3,
        )
    elif optim == "adamW":
        optimizer = torch.optim.AdamW(
            (param for param in model.parameters() if param.requires_grad),
            lr=lr,
            weight_decay=1e-2,
        )

    # -------------- select scheduler -------------- #
    best_error = None
    if scheduler_type == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=end_lr)

    elif scheduler_type == "StepLR":
        scheduler = StepLR(
            optimizer, np.round(num_epoch / (1 + np.log10(lr / end_lr))), 0.1
        )
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

    # -------------- train & valid -------------- #
    for epoch in range(num_epoch):
        if (
            "epoch_stop" in cfg["train_cfg"].keys()
            and epoch > cfg["train_cfg"]["epoch_stop"]
        ):
            break

        # -------------- train phase
        model.train()
        T_max = cfg["loss_cfg"]["distrill_response_temperature"]
        temperature = T_max

        train_losses = {"total": 0}
        logging.info(
            "epoch: {}, lr:{:.8f}, temperature:{:.4f}".format(
                epoch, optimizer.state_dict()["param_groups"][0]["lr"], temperature
            )
        )

        pbar = tqdm(train_dataloader, desc=f"epoch:{epoch}, train")
        for (
            fp,
            patch,
            cap,
            _,
            target,
            target_prob_x,
            target_prob_y,
            target_prob_theta,
        ) in pbar:
            cap = cap.float().to(device)
            patch = patch.float().to(device)
            fp = fp.float().to(device)
            with torch.no_grad():
                [feat_t, response_xy_t, response_theta_t] = teacher_model(
                    fp, need_feat=True, need_response=True, need_res=False
                )

            (
                [pred_xy, pred_theta],
                [pred_xy_double, pred_theta_double],
                [pred_xy_main, pred_theta_main],
                [pred_xy_aux, pred_theta_aux],
                [response_xy_s, response_theta_s],
                [feat_t, feat_s],
            ) = model([patch, cap, feat_t])

            target = target.float().to(device)
            target_prob_x = target_prob_x.float().to(device)
            target_prob_y = target_prob_y.float().to(device)
            target_prob_theta = target_prob_theta.float().to(device)

            vec_xy = classify2vector_trans(
                pred_xy, out_form=trans_out_form, trans_num_classes=trans_num_classes
            )
            vec_theta = classify2vector_rot(
                pred_theta, out_form=rot_out_form, rot_num_classes=rot_num_classes
            )

            loss, items = criterion(
                pred_xy,
                pred_theta,
                vec_xy,
                vec_theta,
                target,
                target_prob_x,
                target_prob_y,
                target_prob_theta,
            )

            loss_scale = loss.item()

            loss_double, _ = criterion(
                pred_xy_double,
                pred_theta_double,
                vec_xy,
                vec_theta,
                target,
                target_prob_x,
                target_prob_y,
                target_prob_theta,
            )
            loss_main, _ = criterion(
                pred_xy_main,
                pred_theta_main,
                vec_xy,
                vec_theta,
                target,
                target_prob_x,
                target_prob_y,
                target_prob_theta,
            )
            loss_aux, _ = criterion(
                pred_xy_aux,
                pred_theta_aux,
                vec_xy,
                vec_theta,
                target,
                target_prob_x,
                target_prob_y,
                target_prob_theta,
            )

            loss += 0.2 * loss_main + 0.2 * loss_aux + 0.4 * loss_double

            if epoch + 1 > 0:  # for which start KD
                if "Response" == args.distrill_mode:
                    loss_distrill_response = distrill_criterions_dict[
                        "distrill_Response"
                    ](response_theta_t, response_theta_s, temperature)
                    c = response_xy_s.shape[1]
                    loss_distrill_response += distrill_criterions_dict[
                        "distrill_Response"
                    ](
                        response_xy_t[:, : c // 2],
                        response_xy_s[:, : c // 2],
                        temperature,
                    )
                    loss_distrill_response += distrill_criterions_dict[
                        "distrill_Response"
                    ](
                        response_xy_t[:, c // 2 :],
                        response_xy_s[:, c // 2 :],
                        temperature,
                    )
                    loss_distrill_response *= 1.0
                    items["distrill_Response"] = loss_distrill_response.item()
                    loss += loss_distrill_response

                if "Relation" == args.distrill_mode:
                    loss_distrill_relation = distrill_criterions_dict[
                        "distrill_Relation"
                    ](feat_t, feat_s)
                    loss_distrill_relation *= 10.0
                    items["distrill_Relation"] = loss_distrill_relation.item()
                    loss += loss_distrill_relation

                if "Feature" == args.distrill_mode:
                    loss_distrill_feature = distrill_criterions_dict[
                        "distrill_Feature"
                    ](feat_t, feat_s)
                    loss_distrill_feature *= 10.0
                    items["distrill_Feature"] = loss_distrill_feature.item()
                    loss += loss_distrill_feature

                if "RkdDistance" == args.distrill_mode:
                    loss_distrill_relation_RKD = distrill_criterions_dict[
                        "distrill_RkdDistance"
                    ](feat_t, feat_s)
                    loss_distrill_relation_RKD *= 10.0
                    items["distrill_RkdDistance"] = loss_distrill_relation_RKD.item()
                    loss += loss_distrill_relation_RKD

                if "CLIP" == args.distrill_mode:
                    loss_distrill_clip = distrill_criterions_dict["distrill_CLIP"](
                        feat_t, feat_s
                    )

                    loss_distrill_clip *= 1.0
                    items["distrill_CLIP"] = loss_distrill_clip.item()
                    loss += loss_distrill_clip

            klist = items.keys()
            train_losses["total"] += loss.item() / len(train_dataloader)
            for k in klist:
                if k in train_losses:
                    train_losses[k] += items[k] / len(train_dataloader)
                else:
                    train_losses[k] = items[k] / len(train_dataloader)

            pbar.set_postfix(**{"loss": loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del loss

        pbar.close()

        klist = train_losses.keys()
        logging_info = "\tTRAIN: ".format(epoch)
        for k in klist:
            logging_info = logging_info + "{}:{:.4f}, ".format(k, train_losses[k])
        logging.info(logging_info)

        # -------------- valid phase
        if valid is False:
            continue
        model.eval()
        with torch.no_grad():
            valid_losses = {
                "total": 0,
            }
            pbar = tqdm(valid_dataloader, desc=f"epoch:{epoch}, val")

            for (
                fp,
                patch,
                cap,
                _,
                target,
                target_prob_x,
                target_prob_y,
                target_prob_theta,
            ) in pbar:
                cap = cap.float().to(device)
                patch = patch.float().to(device)

                [pred_xy, pred_theta] = model([patch, cap])

                target = target.float().to(device)
                target_prob_x = target_prob_x.float().to(device)
                target_prob_y = target_prob_y.float().to(device)
                target_prob_theta = target_prob_theta.float().to(device)

                vec_xy = classify2vector_trans(
                    pred_xy,
                    out_form=trans_out_form,
                    trans_num_classes=trans_num_classes,
                )
                vec_theta = classify2vector_rot(
                    pred_theta, out_form=rot_out_form, rot_num_classes=rot_num_classes
                )

                valid_loss, items = criterion(
                    pred_xy,
                    pred_theta,
                    vec_xy,
                    vec_theta,
                    target,
                    target_prob_x,
                    target_prob_y,
                    target_prob_theta,
                )

                klist = items.keys()
                valid_losses["total"] += valid_loss.item() / len(valid_dataloader)
                for k in klist:
                    if k in valid_losses:
                        valid_losses[k] += items[k] / len(valid_dataloader)
                    else:
                        valid_losses[k] = items[k] / len(valid_dataloader)

                pbar.set_postfix(**{"loss": valid_loss.item()})

                items = criterion_eval(vec_xy, vec_theta, target)
                klist = items.keys()
                for k in klist:
                    if k in valid_losses:
                        valid_losses[k] += items[k] / len(valid_dataloader)
                    else:
                        valid_losses[k] = items[k] / len(valid_dataloader)

            pbar.close()

            klist = valid_losses.keys()
            logging_info = "\tVALID: ".format(epoch)
            for k in klist:
                logging_info = logging_info + "{}:{:.4f}, ".format(k, valid_losses[k])
            logging.info(logging_info)

        # -------------- scheduler
        if scheduler_type == "ReduceLROnPlateau":
            scheduler.step(valid_losses["total"])
        else:
            scheduler.step()

        # save
        if save_dir is not None:
            if (not np.isnan(valid_losses["valid-rot"])) and (
                ((best_error is None)) or (valid_losses["valid-rot"] < best_error)
            ):
                best_error = valid_losses["valid-rot"]
                save_model(model, osp.join(save_dir, f"best.pth"))
                logging.info("SAVE BEST MODEL!")

            if scheduler_type == "ReduceLROnPlateau":
                if optimizer.state_dict()["param_groups"][0]["lr"] < end_lr:
                    return
            elif epoch >= save_checkpoint:
                save_model(model, osp.join(save_dir, f"epoch_{epoch}.pth"))

    return


if __name__ == "__main__":
    set_seed(seed=7)

    args = get_args()

    # your path to teach model weight
    TEACHER_MODEL_DIR_DICT = {
        "fp-rot45": "/your_path/rot45/Single-fp",
        "fp-rot90": "/your_path/rot90/Single-fp",
        "fp-rot135": "/your_path/rot135/Single-fp",
        "fp-rot180": "/your_path/rot180/Single-fp",
    }

    config_path = f"./configs/config_{args.config_name}.yaml"
    with open(config_path, "r") as config:
        cfg = yaml.safe_load(config)

    cuda_ids = args.cuda_ids
    if "," in cuda_ids:
        cuda_ids = [int(x) for x in args.cuda_ids.split(",")]
    else:
        cuda_ids = [int(cuda_ids)]
    cfg["train_cfg"]["cuda_ids"] = cuda_ids
    cfg["train_cfg"]["batch_size"] = args.batch_size
    cfg["model_cfg"]["trans_out_form"] = args.trans_out_form
    cfg["model_cfg"]["trans_num_classes"] = args.trans_num_classes
    cfg["model_cfg"]["rot_out_form"] = args.rot_out_form
    cfg["model_cfg"]["rot_num_classes"] = args.rot_num_classes
    cfg["loss_cfg"]["supervise_mode"] = args.supervise_mode
    cfg["loss_cfg"]["trans_loss_form"] = args.trans_loss_form
    cfg["loss_cfg"]["rot_loss_form"] = args.rot_loss_form
    cfg["loss_cfg"]["trans_loss_weight"] = args.trans_loss_weight
    cfg["loss_cfg"]["distrill_mode"] = args.distrill_mode
    cfg["loss_cfg"][
        "distrill_response_temperature"
    ] = args.distrill_response_temperature

    # set save dir
    save_basedir = cfg["save_cfg"]["save_basedir"]
    model_name = cfg["model_cfg"]["model_name"]
    inp_mode = cfg["model_cfg"]["inp_mode"]
    trans_out_form = cfg["model_cfg"]["trans_out_form"]
    trans_num_classes = cfg["model_cfg"]["trans_num_classes"]
    rot_out_form = cfg["model_cfg"]["rot_out_form"]
    rot_num_classes = cfg["model_cfg"]["rot_num_classes"]
    supervise_mode = cfg["loss_cfg"]["supervise_mode"]
    trans_loss_form = cfg["loss_cfg"]["trans_loss_form"]
    rot_loss_form = cfg["loss_cfg"]["rot_loss_form"]
    config_name = args.config_name

    save_dirname = f"{model_name}-{inp_mode}-S{supervise_mode}-TF{trans_out_form}{trans_num_classes}-RF{rot_out_form}{rot_num_classes}-TL{trans_loss_form}-RL{rot_loss_form}"
    if cfg["save_cfg"]["save_title"] == "time":
        now = datetime.datetime.now()
        save_dir = osp.join(
            save_basedir, save_dirname, now.strftime("%Y-%m-%d-%H-%M-%S")
        )
    else:
        save_dir = osp.join(save_basedir, save_dirname, cfg["save_cfg"]["save_title"])

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    with open(osp.join(save_dir, "config.yaml"), "w") as file:
        yaml.dump(cfg, file, default_flow_style=False)

    # set database
    train_info_path = cfg["db_cfg"]["train_info_path"]
    valid_info_path = cfg["db_cfg"]["valid_info_path"]
    train_info = np.load(train_info_path, allow_pickle=True).item()
    valid_info = np.load(valid_info_path, allow_pickle=True).item()

    # logging losses
    logging_path = osp.join(save_dir, "info.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        filename=logging_path,
        filemode="w",
    )

    # set dataloader
    train_loader = get_dataloader_train(
        fp_lst=train_info["fp_bimg_lst"],
        patch_lst=None,
        mask_lst=train_info["seg_lst"],
        cap_lst=None,
        pose_lst=train_info["pose_lst"],
        inp_mode=cfg["model_cfg"]["inp_mode"],
        inp_patch_size=cfg["model_cfg"]["inp_patch_size"],
        trans_num_classes=cfg["model_cfg"]["trans_num_classes"],
        rot_num_classes=cfg["model_cfg"]["rot_num_classes"],
        apply_aug=cfg["train_cfg"]["apply_aug"],
        trans_aug=cfg["train_cfg"]["trans_aug"],
        rot_aug=cfg["train_cfg"]["rot_aug"],
        batch_size=cfg["train_cfg"]["batch_size"],
        shuffle=True,
    )

    valid_loader = get_dataloader_valid(
        fp_lst=valid_info["fp_bimg_lst"],
        patch_lst=valid_info["patch_bimg_lst"],
        mask_lst=valid_info["seg_lst"],
        cap_lst=valid_info["cap_lst"],
        pose_lst=valid_info["pose_lst"],
        inp_mode=cfg["model_cfg"]["inp_mode"],
        inp_patch_size=cfg["model_cfg"]["inp_patch_size"],
        trans_num_classes=cfg["model_cfg"]["trans_num_classes"],
        rot_num_classes=cfg["model_cfg"]["rot_num_classes"],
        apply_aug=False,
        batch_size=128,
        shuffle=False,
    )

    # set models
    device = torch.device(
        "cuda:{}".format(str(cfg["train_cfg"]["cuda_ids"][0]))
        if torch.cuda.is_available()
        else "cpu"
    )

    if cfg["model_cfg"]["model_name"] == "DRACO":
        model = DRACO(
            inp_mode=cfg["model_cfg"]["inp_mode"],
            trans_out_form=cfg["model_cfg"]["trans_out_form"],
            trans_num_classes=cfg["model_cfg"]["trans_num_classes"],
            rot_out_form=cfg["model_cfg"]["rot_out_form"],
            rot_num_classes=cfg["model_cfg"]["rot_num_classes"],
        )

    logging.info("Model: {}".format(cfg["model_cfg"]["model_name"]))

    model = torch.nn.DataParallel(
        model,
        device_ids=cfg["train_cfg"]["cuda_ids"],
        output_device=cfg["train_cfg"]["cuda_ids"][0],
    ).to(device)

    # --- teacher model
    teacher_model = DRACO_Single(
        inp_mode=cfg["model_cfg"]["inp_mode"],
        trans_out_form=cfg["model_cfg"]["trans_out_form"],
        trans_num_classes=cfg["model_cfg"]["trans_num_classes"],
        rot_out_form=cfg["model_cfg"]["rot_out_form"],
        rot_num_classes=cfg["model_cfg"]["rot_num_classes"],
    )

    pth_path = TEACHER_MODEL_DIR_DICT[
        "fp-rot{}".format(str(cfg["train_cfg"]["rot_aug"]))
    ]
    pth_path = osp.join(pth_path, "best.pth")
    teacher_model.load_state_dict(
        torch.load(pth_path, map_location=f"cuda:{cuda_ids[0]}")
    )
    teacher_model = torch.nn.DataParallel(
        teacher_model,
        device_ids=cuda_ids,
        output_device=cuda_ids[0],
    ).to(device)
    teacher_model.eval()

    logging.info("******** begin training ********")
    train(
        model=model,
        teacher_model=teacher_model,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        device=device,
        cfg=cfg,
        save_dir=save_dir,
        save_checkpoint=cfg["train_cfg"]["epochs"] - cfg["train_cfg"]["ckpts_num"],
    )
