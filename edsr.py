import math

import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class MeanShift(nn.Conv2d):
    def __init__(
        self,
        rgb_range=255,
        rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1.0, 1.0, 1.0),
        sign=-1,
    ):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, res_scale=1):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            default_conv(n_feats, n_feats, kernel_size),
            nn.ReLU(True),
            default_conv(n_feats, n_feats, kernel_size),
        )
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class EDSRUpsampler(nn.Sequential):
    def __init__(self, scale, n_feats):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(default_conv(n_feats, 4 * n_feats, 3))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(default_conv(n_feats, 9 * n_feats, 3))
            m.append(nn.PixelShuffle(3))

        super(EDSRUpsampler, self).__init__(*m)


class EDSR(nn.Module):
    def __init__(
        self,
        scale,
        n_resblocks,
        n_feats,
        n_colors=3,
        res_scale=1,
        rgb_range=255,
        rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1, 1, 1),
    ):
        super(EDSR, self).__init__()
        kernel_size = 3
        self.sub_mean = MeanShift(
            sign=-1, rgb_range=rgb_range, rgb_mean=rgb_mean, rgb_std=rgb_std
        )
        self.add_mean = MeanShift(
            sign=1, rgb_range=rgb_range, rgb_mean=rgb_mean, rgb_std=rgb_std
        )

        # define head module
        m_head = [default_conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(n_feats, kernel_size, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(default_conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            EDSRUpsampler(scale, n_feats),
            default_conv(n_feats, n_colors, kernel_size),
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x
