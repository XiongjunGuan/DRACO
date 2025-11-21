"""
Description:
Author: Xiongjun Guan
Date: 2024-06-04 15:50:36
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2025-02-25 23:23:28

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-6


class CELoss(nn.Module):

    def __init__(self, need_softmax=False, need_log=True):
        super().__init__()
        self.need_softmax = need_softmax
        self.need_log = need_log

    def forward(self, pred, target):
        if self.need_softmax:
            pred = F.softmax(pred, dim=1)
        if self.need_log:
            pred = pred.clamp_min(1e-6).log()
        loss = torch.mean(-torch.sum(target * pred, dim=1))
        return loss


class QualityCELoss(nn.Module):

    def __init__(self, need_softmax=False, need_log=True):
        super().__init__()
        self.need_softmax = need_softmax
        self.need_log = need_log

    def forward(self, pred, target):
        if self.need_softmax:
            pred = F.softmax(pred, dim=1)
        assert self.need_log
        if self.need_log:
            log_pred = pred.log()
        loss = torch.mean(
            -torch.sum(torch.abs(pred - target) * target * log_pred, dim=1)
        )
        return loss


class JSDivLoss(nn.Module):

    def __init__(self, need_softmax=False):
        super().__init__()
        self.loss_func = nn.KLDivLoss()
        self.need_softmax = need_softmax

    def forward(self, p_output, q_output):
        if self.need_softmax:
            p_output = F.softmax(p_output)
            q_output = F.softmax(q_output)
        log_mean_output = ((p_output + q_output) / 2).log()
        return (
            self.loss_func(log_mean_output, p_output)
            + self.loss_func(log_mean_output, q_output)
        ) / 2


class FinalLoss(torch.nn.Module):

    def __init__(
        self,
        supervise_mode="rot",
        trans_loss_form="mse",
        rot_out_form="claSum",
        rot_loss_form="mse",
        trans_loss_weight=0.2,
    ):
        super().__init__()
        self.supervise_mode = supervise_mode
        self.trans_loss_form = trans_loss_form
        self.rot_loss_form = rot_loss_form
        self.trans_loss_weight = trans_loss_weight

        if self.trans_loss_form == "mse":
            self.trans_func = nn.MSELoss()
        elif self.trans_loss_form == "SmoothL1":
            self.trans_func = nn.SmoothL1Loss()
        elif self.trans_loss_form == "L1":
            self.trans_func = nn.L1Loss()
        elif self.trans_loss_form == "CE":
            self.trans_func = CELoss()
        elif self.trans_loss_form == "JS":
            self.trans_func = JSDivLoss()

        if self.rot_loss_form in ["mse_ang", "mse_tan"]:
            self.rot_func = nn.MSELoss()
        elif self.rot_loss_form in ["SmoothL1_ang", "SmoothL1_tan"]:
            self.rot_func = nn.SmoothL1Loss()
        elif self.rot_loss_form in ["L1_ang", "L1_tan"]:
            self.rot_func = nn.L1Loss()
        elif self.rot_loss_form == "CE":
            self.rot_func = CELoss()
        elif self.rot_loss_form == "JS":
            self.rot_func = JSDivLoss()

    def forward(
        self,
        pred_xy,
        pred_theta,
        vec_xy,
        vec_theta,
        vec_target,
        target_prob_x,
        target_prob_y,
        target_prob_theta,
    ):

        loss_items = {}
        loss = 0

        if "rot" in self.supervise_mode:
            if self.rot_loss_form in [
                "mse_ang",
                "SmoothL1_ang",
                "L1_ang",
                "mse_tan",
                "SmoothL1_tan",
                "L1_tan",
            ]:
                if self.rot_loss_form.split("_")[1] == "tan":
                    loss_cos = self.rot_func(vec_theta[:, -3], vec_target[:, -3])
                    loss_sin = self.rot_func(vec_theta[:, -2], vec_target[:, -2])
                    loss_items["theta-cos"] = loss_cos.item()
                    loss_items["theta-sin"] = loss_sin.item()
                    loss += loss_cos + loss_sin
                elif self.rot_loss_form.split("_")[1] == "ang":
                    rot_const = 90  # [-180,180] -> [-2,2]
                    dtheta = torch.abs(vec_theta[:, -1] - vec_target[:, -1])
                    dtheta = torch.min(dtheta, 360 - dtheta) / rot_const

                    loss_theta = self.rot_func(
                        dtheta, torch.zeros_like(dtheta).to(dtheta.device)
                    )
                    loss_items["theta"] = loss_theta.item()
                    loss += loss_theta
            elif self.rot_loss_form in ["CE", "JS"]:
                loss_theta = self.rot_func(pred_theta, target_prob_theta)
                loss_items["theta"] = loss_theta.item()
                loss += loss_theta

        if "trans" in self.supervise_mode:
            if self.trans_loss_form in ["mse", "SmoothL1", "L1"]:
                trans_const = 64  # [-256, 256] -> [-4, 4]
                loss_x = self.trans_func(
                    vec_xy[:, 0] / trans_const, vec_target[:, 0] / trans_const
                )
                loss_y = self.trans_func(
                    vec_xy[:, 1] / trans_const, vec_target[:, 1] / trans_const
                )

                loss_items["pos-x"] = self.trans_loss_weight * loss_x.item()
                loss_items["pos-y"] = self.trans_loss_weight * loss_y.item()
                loss += loss_x + loss_y
            elif self.trans_loss_form in ["CE", "JS"]:
                _, c = pred_xy.shape[:2]
                loss_x = self.trans_func(pred_xy[:, : c // 2], target_prob_x)
                loss_y = self.trans_func(pred_xy[:, c // 2 :], target_prob_y)
                loss_items["pos-x"] = self.trans_loss_weight * loss_x.item()
                loss_items["pos-y"] = self.trans_loss_weight * loss_y.item()
                loss += loss_x + loss_y

        return loss, loss_items


class EvalLoss(torch.nn.Module):

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, vec_xy, vec_theta, vec_target):
        loss_items = {}

        dx = vec_xy[:, 0] - vec_target[:, 0]
        dy = vec_xy[:, 1] - vec_target[:, 1]
        loss_x = torch.mean(torch.abs(dx))
        loss_y = torch.mean(torch.abs(dy))
        loss_dis = torch.mean(torch.sqrt(torch.square(dx) + torch.square(dy)))
        loss_items["valid-x"] = loss_x.item()
        loss_items["valid-y"] = loss_y.item()
        loss_items["valid-trans"] = loss_dis.item()

        dtheta = torch.abs(vec_theta[:, -1] - vec_target[:, -1])
        dtheta = torch.min(dtheta, 360 - dtheta)
        loss_theta = torch.mean(dtheta)
        loss_items["valid-rot"] = loss_theta.item()

        return loss_items


class DistillationLoss_Feature(torch.nn.Module):

    def __init__(
        self,
    ):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, feat_t, feat_s):

        loss = self.criterion(feat_t.detach(), feat_s)
        return loss


class DistillationLoss_Response(torch.nn.Module):

    def __init__(self, T=10):
        super().__init__()
        self.T = T
        self.criterion = CELoss(need_softmax=False, need_log=False)

    def forward(self, y_t, y_s, T=None):
        if T is None:
            T = self.T
        p_s = F.log_softmax(y_s / T, dim=1)
        p_t = F.softmax(y_t / T, dim=1).detach()
        loss = self.criterion(p_s, p_t)
        return loss


class DistillationLoss_CLIP(torch.nn.Module):

    def __init__(
        self,
    ):
        super().__init__()
        self.temperature = 0.07

    def forward(self, ft_proj, fs_proj):
        batch_size = ft_proj.size(0)

        ft_proj = F.normalize(ft_proj, dim=1)
        fs_proj = F.normalize(fs_proj, dim=1)

        # 计算对比损失（对称结构）
        logits = (fs_proj @ ft_proj.T) / self.temperature

        # 构造标签矩阵
        labels = torch.arange(batch_size, device=ft_proj.device).detach()

        # 交叉熵损失
        loss = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )

        return loss


def compute_crosslogitdist_loss(logit_stu, logit_tea):
    assert logit_stu.shape == logit_tea.shape
    assert len(logit_stu.shape) == 2

    distil_loss_fn = torch.nn.MSELoss()
    logit_norm_fn_stu = F.normalize
    logit_norm_fn_tea = F.normalize

    logit_stu_normalized_mapped = logit_norm_fn_stu(logit_stu, dim=1)
    logit_tea_normalized_mapped = logit_norm_fn_tea(logit_tea, dim=1)

    diffmat_ss = logit_stu_normalized_mapped.unsqueeze(
        0
    ) - logit_stu_normalized_mapped.unsqueeze(1)
    distmat_ss = torch.norm(diffmat_ss, p=2, dim=-1)
    diffmat_st = logit_stu_normalized_mapped.unsqueeze(
        0
    ) - logit_tea_normalized_mapped.unsqueeze(1)
    distmat_st = torch.norm(diffmat_st, p=2, dim=-1)
    diffmat_tt = logit_tea_normalized_mapped.unsqueeze(
        0
    ) - logit_tea_normalized_mapped.unsqueeze(1)
    distmat_tt = torch.norm(diffmat_tt, p=2, dim=-1)

    distil_loss_st2ss = distil_loss_fn(distmat_ss, distmat_st.detach())
    distil_loss_tt2ss = distil_loss_fn(distmat_ss, distmat_tt.detach())

    return distil_loss_st2ss, distil_loss_tt2ss


def compute_crosslogitsim_loss(logit_stu, logit_tea):
    assert logit_stu.shape == logit_tea.shape
    assert len(logit_stu.shape) == 2

    distil_loss_fn = torch.nn.MSELoss()
    logit_norm_fn_stu = F.normalize
    logit_norm_fn_tea = F.normalize

    logit_stu_normalized_mapped = logit_norm_fn_stu(logit_stu, dim=1)
    logit_tea_normalized_mapped = logit_norm_fn_tea(logit_tea, dim=1)

    norm_s = torch.norm(logit_stu_normalized_mapped, p=2, dim=1, keepdim=True)  # [b,1]
    normsqmat_ss = norm_s @ norm_s.T  # [b,b]
    norm_t = torch.norm(logit_tea_normalized_mapped, p=2, dim=1, keepdim=True)
    normsqmat_tt = norm_t @ norm_t.T
    normsqmat_st = norm_s @ norm_t.T

    simmat_ss = logit_stu_normalized_mapped @ logit_stu_normalized_mapped.T
    simmat_st = logit_stu_normalized_mapped @ logit_tea_normalized_mapped.T
    simmat_tt = logit_tea_normalized_mapped @ logit_tea_normalized_mapped.T

    simmat_ss = simmat_ss / normsqmat_ss
    simmat_st = simmat_st / normsqmat_st
    simmat_tt = simmat_tt / normsqmat_tt

    distil_loss_st2ss = distil_loss_fn(simmat_ss, simmat_st.detach())
    distil_loss_tt2ss = distil_loss_fn(simmat_ss, simmat_tt.detach())

    return distil_loss_st2ss, distil_loss_tt2ss


class DistillationLoss_Relation(torch.nn.Module):
    """https://github.com/sijieaaa/DistilVPR/

    Args:
        torch (_type_): _description_
    """

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, logit_tea, logit_stu):
        crosslogitdistloss_st2ss, crosslogitdistloss_tt2ss = (
            compute_crosslogitdist_loss(logit_stu, logit_tea)
        )
        crosslogitsimloss_st2ss, crosslogitsimloss_tt2ss = compute_crosslogitsim_loss(
            logit_stu, logit_tea
        )
        loss = (
            crosslogitdistloss_st2ss
            + crosslogitdistloss_tt2ss
            + crosslogitsimloss_st2ss
            + crosslogitsimloss_tt2ss
        )
        return loss


class RkdDistance(nn.Module):

    def forward(
        self,
        tea,
        stu,
        squared=False,
        eps=1e-12,
        distance_weight=1,
        angle_weight=1,
    ):
        # RKD distance loss
        with torch.no_grad():
            t_d = _pdist(tea, squared, eps)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = _pdist(stu, squared, eps)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        # RKD Angle loss
        with torch.no_grad():
            td = tea.unsqueeze(0) - tea.unsqueeze(1)
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = stu.unsqueeze(0) - stu.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        loss = distance_weight * loss_d + angle_weight * loss_a

        return loss


def _pdist(e, squared, eps):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res
