import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.rrdbnet_arch import RRDB
from .arch_util import make_layer, pixel_unshuffle


class AFT(nn.Module):
    def __init__(self, is_adapt=True):
        super(AFT, self).__init__()

        self.gamma_conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.gamma_bn1 = nn.BatchNorm2d(64)
        self.gamma_conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.gamma_bn2 = nn.BatchNorm2d(64)

        self.beta_conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.beta_bn1 = nn.BatchNorm2d(64)
        self.beta_conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.beta_bn2 = nn.BatchNorm2d(64)

        self.w_conv1 = nn.Conv2d(128, 64, 3, 1, 1)

        self.conv1x1 = nn.Conv2d(128, 64, 1, 1)
        self.bn1x1 = nn.BatchNorm2d(64)

        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU()

        self.is_adapt = is_adapt

    def forward(self, x, conditions):

        gamma = self.lrelu(self.gamma_bn1(self.gamma_conv1(conditions)))
        gamma = self.lrelu(self.gamma_bn2(self.gamma_conv2(gamma)))

        beta = self.lrelu(self.beta_bn1(self.beta_conv1(conditions)))
        beta = self.lrelu(self.beta_bn2(self.beta_conv1(beta)))

        sft = x * gamma + beta

        if self.is_adapt:
            weight_map = self.sigmoid(self.w_conv1(torch.cat((x, sft), 1)))
            out = self.relu(self.bn1x1(self.conv1x1(torch.cat((x, sft*weight_map), 1))))
        else:
            out = sft
        return out


class ATB(nn.Module):
    def __init__(self, scale: int = 8):
        super(ATB, self).__init__()

        self.scale = scale

        self.aft1 = AFT()
        self.aft2 = AFT()

        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)

        convs = []
        bns = []
        for i in range(self.scale):
            convs.append(nn.Conv2d(8, 8, 3, 1, 1))
            bns.append(nn.BatchNorm2d(8))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = nn.ReLU()

        self.conv1x1 = nn.Conv2d(64, 64, 1, 1)
        self.bn1x1 = nn.BatchNorm2d(64)

    def forward(self, x, conditions):

        residual = x

        out = self.aft1(x, conditions)
        out = self.bn1(self.conv1(out))
        out = self.aft2(out, conditions)

        spx = torch.split(out, 8, 1)
        for i in range(self.scale):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = self.bn1x1(self.conv1x1(out))
        out = out + residual
        out = self.relu(out)

        return out


class RATB(nn.Module):
    def __init__(self):
        super(RATB, self).__init__()

        self.atb1 = ATB()
        self.atb2 = ATB()
        self.atb3 = ATB()

    def forward(self, x, conditions):

        residual = x
        out = self.atb1(x, conditions)
        out = self.atb2(out, conditions)
        out = self.atb3(out, conditions)

        out = out + residual
        return out


@ARCH_REGISTRY.register()
class MFNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(MFNet, self).__init__()

        self.ratb1 = RATB()
        self.ratb2 = RATB()
        self.aft1 = AFT()

        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.RRDB = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_cond = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.bn = nn.BatchNorm2d(64)

        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, c):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
            condition = pixel_unshuffle(c, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
            condition = pixel_unshuffle(c, scale=4)
        else:
            feat = x
            condition = c

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.RRDB(feat))

        condition = self.bn(self.conv_first(condition))
        condition = self.bn(self.conv_cond(condition))
        condition = self.bn(self.conv_cond(condition))
        condition = self.bn(self.conv_cond(condition))

        feat = self.ratb1(feat, condition)
        feat = self.ratb2(feat, condition)
        feat = self.aft1(feat, condition)

        feat = feat + body_feat

        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out