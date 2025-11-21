import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .CBAM import CBAM
from .ResMLP import ResMLP
from .resnext import ResNextBlock


class NormalizeModule(nn.Module):

    def __init__(self, m0=0, var0=1, eps=1e-6):
        super(NormalizeModule, self).__init__()
        self.m0 = m0
        self.var0 = var0
        self.eps = eps

    def forward(self, x):
        x_m = x.mean(dim=(1, 2, 3), keepdim=True)
        x_var = x.var(dim=(1, 2, 3), keepdim=True)
        y = (self.var0 * (x - x_m) ** 2 / x_var.clamp_min(self.eps)).sqrt()
        y = torch.where(x > x_m, self.m0 + y, self.m0 - y)
        return y


class ConvBnPRelu(nn.Module):

    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chn,
            out_chn,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm2d(out_chn, eps=0.001, momentum=0.99)
        self.relu = nn.PReLU(out_chn, init=0)

    def forward(self, input):
        y = self.conv(input)
        y = self.bn(y)
        y = self.relu(y)
        return y


class DRACO_Double(nn.Module):

    def __init__(
        self,
        inp_mode="patch_cap",
        trans_out_form="claSum",
        trans_num_classes=120,
        rot_out_form="claSum",
        rot_num_classes=120,
        main_channel_lst=[64, 128, 256, 512, 1024],
        layer_lst=[3, 4, 6, 3],
        aux_channel_lst=[32, 64, 128, 256, 512],
    ):
        super(DRACO_Double, self).__init__()
        self.trans_out_form = trans_out_form
        self.rot_out_form = rot_out_form

        self.norm_layer = NormalizeModule(m0=0, var0=1)

        self.main_layer1 = nn.Sequential(
            ConvBnPRelu(1, main_channel_lst[0], 7, stride=2, padding=3),
            ConvBnPRelu(main_channel_lst[0], main_channel_lst[0], 3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        if inp_mode == "patch_cap":
            self.aux_layer1 = nn.Sequential(
                ConvBnPRelu(1, aux_channel_lst[0], 3),
                ConvBnPRelu(aux_channel_lst[0], aux_channel_lst[0], 3),
            )
        else:
            self.aux_layer1 = nn.Sequential(
                ConvBnPRelu(1, aux_channel_lst[0], 7, stride=2, padding=3),
                ConvBnPRelu(aux_channel_lst[0], aux_channel_lst[0], 3),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        self.main_layer2 = self._make_layers(
            ResNextBlock,
            in_channels=main_channel_lst[0],
            out_channels=main_channel_lst[1],
            groups=32,
            stride=2,
            num_layers=layer_lst[0],
        )
        self.aux_layer2 = self._make_layers(
            ResNextBlock,
            in_channels=aux_channel_lst[0],
            out_channels=aux_channel_lst[1],
            groups=32,
            stride=1 if inp_mode == "patch_cap" else 2,
            num_layers=layer_lst[0],
        )

        self.main_layer3 = self._make_layers(
            ResNextBlock,
            in_channels=main_channel_lst[1],
            out_channels=main_channel_lst[2],
            groups=32,
            stride=2,
            num_layers=layer_lst[1],
        )
        self.main_att3 = CBAM(main_channel_lst[2])

        self.aux_layer3 = self._make_layers(
            ResNextBlock,
            in_channels=aux_channel_lst[1],
            out_channels=aux_channel_lst[2],
            groups=32,
            stride=1 if inp_mode == "patch_cap" else 2,
            num_layers=layer_lst[1],
        )
        self.aux_att3 = CBAM(aux_channel_lst[2])

        self.main_layer4 = self._make_layers(
            ResNextBlock,
            in_channels=main_channel_lst[2],
            out_channels=main_channel_lst[3],
            groups=32,
            stride=2,
            num_layers=layer_lst[2],
        )
        self.main_att4 = CBAM(main_channel_lst[3])

        self.aux_layer4 = self._make_layers(
            ResNextBlock,
            in_channels=aux_channel_lst[2],
            out_channels=aux_channel_lst[3],
            groups=32,
            stride=1 if inp_mode == "patch_cap" else 2,
            num_layers=layer_lst[2],
        )
        self.aux_att4 = CBAM(aux_channel_lst[3])

        self.main_layer5 = self._make_layers(
            ResNextBlock,
            in_channels=main_channel_lst[3],
            out_channels=main_channel_lst[4],
            groups=32,
            stride=2,
            num_layers=layer_lst[3],
        )
        self.main_att5 = CBAM(main_channel_lst[4])

        self.aux_layer5 = self._make_layers(
            ResNextBlock,
            in_channels=aux_channel_lst[3],
            out_channels=aux_channel_lst[4],
            groups=32,
            stride=1 if inp_mode == "patch_cap" else 2,
            num_layers=layer_lst[3],
        )
        self.aux_att5 = CBAM(aux_channel_lst[4])

        self.avgpool_flatten_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1)
        )

        # --- mapping
        dim = 128
        hidden_dim = dim * 4
        depth = 4
        self.main_map_layer = nn.Sequential(
            nn.Linear(main_channel_lst[4], dim),
            nn.GELU(),
            ResMLP(dim, hidden_dim, depth=depth),
        )

        self.aux_map_layer = nn.Sequential(
            nn.Linear(aux_channel_lst[4], dim),
            nn.GELU(),
            ResMLP(dim, hidden_dim, depth=depth),
        )
        double_dim = main_channel_lst[4] + aux_channel_lst[4]
        self.double_map_layer = nn.Sequential(
            nn.Linear(double_dim, dim),
            nn.GELU(),
            ResMLP(dim, hidden_dim, depth=depth),
        )

        # --- decision fusion for main/aux/double branch
        self.trans_weight_layer = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, 3),
        )
        self.rot_weight_layer = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, 3),
        )

        # --- prediction branch
        if trans_out_form in ["claSum", "claMax"]:
            self.trans_fc_theta = nn.Linear(dim, trans_num_classes)
            self.trans_fc_theta_main = nn.Linear(dim, trans_num_classes)
            self.trans_fc_theta_aux = nn.Linear(dim, trans_num_classes)

        if rot_out_form in ["claSum", "claMax"]:
            self.rot_fc_theta = nn.Linear(dim, rot_num_classes)
            self.rot_fc_theta_main = nn.Linear(dim, rot_num_classes)
            self.rot_fc_theta_aux = nn.Linear(dim, rot_num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(
        self, Block, in_channels, out_channels, groups, stride, num_layers
    ):
        layers = []
        layers.append(
            Block(
                in_channels=in_channels,
                out_channels=out_channels,
                groups=groups,
                stride=stride,
            )
        )
        for _ in range(num_layers - 1):
            layers.append(
                Block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    groups=groups,
                    stride=1,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, inp, feat_only=False):
        [main_inp, aux_inp] = inp
        main_inp = self.norm_layer(main_inp)
        main_feat = self.main_layer1(main_inp)
        main_feat = self.main_layer2(main_feat)
        main_feat = self.main_layer3(main_feat)
        main_feat, _, _ = self.main_att3(main_feat)
        main_feat = self.main_layer4(main_feat)
        main_feat, _, _ = self.main_att4(main_feat)
        main_feat = self.main_layer5(main_feat)
        main_feat, _, _ = self.main_att5(main_feat)

        aux_feat = self.aux_layer1(aux_inp)
        aux_feat = self.aux_layer2(aux_feat)
        aux_feat = self.aux_layer3(aux_feat)
        aux_feat, _, _ = self.aux_att3(aux_feat)
        aux_feat = self.aux_layer4(aux_feat)
        aux_feat, _, _ = self.aux_att4(aux_feat)
        aux_feat = self.aux_layer5(aux_feat)
        aux_feat, _, _ = self.aux_att5(aux_feat)

        main_feat = self.avgpool_flatten_layer(main_feat)
        aux_feat = self.avgpool_flatten_layer(aux_feat)

        double_feat = torch.cat([main_feat, aux_feat], dim=1)

        # --- map
        double_feat = self.double_map_layer(double_feat)
        main_feat = self.main_map_layer(main_feat)
        aux_feat = self.aux_map_layer(aux_feat)

        # --- weight
        uni_feat = torch.cat([double_feat, main_feat, aux_feat], dim=1)
        trans_weight = self.trans_weight_layer(uni_feat)
        rot_weight = self.rot_weight_layer(uni_feat)

        # --- double branch
        pred_xy_double = self.trans_fc_theta(double_feat)

        pred_xy = pred_xy_double * trans_weight[:, 0:1]

        _, c = pred_xy_double.shape
        pred_x_double = F.softmax(pred_xy_double[:, : c // 2], dim=1)
        pred_y_double = F.softmax(pred_xy_double[:, c // 2 :], dim=1)
        pred_xy_double = torch.cat(
            [pred_x_double, pred_y_double], dim=1
        )  # [b, (num_class//2,num_class//2)] for x and y prob

        pred_theta_double = self.rot_fc_theta(double_feat)

        pred_theta = pred_theta_double * rot_weight[:, 0:1]

        pred_theta_double = F.softmax(
            pred_theta_double, dim=1
        )  # [b, num_class] for theta prob

        # --- xy part
        pred_xy_main = self.trans_fc_theta_main(main_feat)
        pred_xy += pred_xy_main * trans_weight[:, 1:2]
        _, c = pred_xy_main.shape
        pred_x_main = F.softmax(pred_xy_main[:, : c // 2], dim=1)
        pred_y_main = F.softmax(pred_xy_main[:, c // 2 :], dim=1)
        pred_xy_main = torch.cat(
            [pred_x_main, pred_y_main], dim=1
        )  # [b, (num_class//2,num_class//2)] for x and y prob

        pred_xy_aux = self.trans_fc_theta_aux(aux_feat)
        pred_xy += pred_xy_aux * trans_weight[:, 2:3]
        _, c = pred_xy_aux.shape
        pred_x_aux = F.softmax(pred_xy_aux[:, : c // 2], dim=1)
        pred_y_aux = F.softmax(pred_xy_aux[:, c // 2 :], dim=1)
        pred_xy_aux = torch.cat(
            [pred_x_aux, pred_y_aux], dim=1
        )  # [b, (num_class//2,num_class//2)] for x and y prob

        # --- theta part
        pred_theta_main = self.rot_fc_theta_main(main_feat)
        pred_theta += pred_theta_main * rot_weight[:, 1:2]
        pred_theta_main = F.softmax(
            pred_theta_main, dim=1
        )  # [b, num_class] for theta prob

        pred_theta_aux = self.rot_fc_theta_aux(aux_feat)
        pred_theta += pred_theta_aux * rot_weight[:, 2:3]
        pred_theta_aux = F.softmax(
            pred_theta_aux, dim=1
        )  # [b, num_class] for theta prob

        _, c = pred_xy.shape
        pred_x = F.softmax(pred_xy[:, : c // 2], dim=1)
        pred_y = F.softmax(pred_xy[:, c // 2 :], dim=1)
        pred_xy = torch.cat(
            [pred_x, pred_y], dim=1
        )  # [b, (num_class//2,num_class//2)] for x and y prob
        pred_theta = F.softmax(pred_theta, dim=1)  # [b, num_class] for theta prob
        if feat_only:
            return [double_feat, main_feat, aux_feat]

        if self.training:
            return (
                [pred_xy, pred_theta],
                [pred_xy_double, pred_theta_double],
                [pred_xy_main, pred_theta_main],
                [pred_xy_aux, pred_theta_aux],
            )
        else:
            return [pred_xy, pred_theta]
