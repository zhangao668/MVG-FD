import timm
from torch import nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=1)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()

        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4
        return ll, lh, hl, hh

class UpSample(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear'):
        super(UpSample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)

class ContrastEnhancedHighFrequency(nn.Module):
    def __init__(self, alpha=1.5):
        super(ContrastEnhancedHighFrequency, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, lh, hl, hh):
        high_freq = self.alpha * (torch.abs(lh) + torch.abs(hl) + torch.abs(hh))
        return high_freq

"""
    Joint Attention module (CA + SA)
"""
class SA(nn.Module):
    def __init__(self, channels):
        super(SA, self).__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.sa(x)
        y = x * out
        return y

class CA(nn.Module):
    def __init__(self,lf=True):
        super(CA, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d(1) if lf else nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        y = self.ap(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x) * self.alpha



class AM(nn.Module):
    def __init__(self, channels,lf):
        super(AM, self).__init__()
        self.CA = CA(lf=lf)
        self.SA = SA(channels)

    def forward(self, x):
        e=x
        x = self.CA(x)
        x = self.SA(x)
        x = e+x
        return x

"""
    High-Frequency Attention Module (HFA)
"""

class RB(nn.Module):
    def __init__(self, channels, lf):
        super(RB, self).__init__()
        self.RB = BasicConv2d(channels, channels, 3, padding=1, bn=nn.InstanceNorm2d if lf else nn.BatchNorm2d)

    def forward(self, x):
        y = self.RB(x)
        return y + x


class ARB(nn.Module):
    def __init__(self, channels, lf):
        super(ARB, self).__init__()
        self.lf = lf
        self.AM = AM(channels, lf)
        self.RB = RB(channels, lf)

        self.mean_conv1 = ConvLayer(1, 16, 1, 1)
        self.mean_conv2 = ConvLayer(16, 16, 3, 1)
        self.mean_conv3 = ConvLayer(16, 1, 1, 1)

        self.std_conv1 = ConvLayer(1, 16, 1, 1)
        self.std_conv2 = ConvLayer(16, 16, 3, 1)
        self.std_conv3 = ConvLayer(16, 1, 1, 1)


    def PONO(self, x, epsilon=1e-5):

        mean = x.mean(dim=1, keepdim=True)

        std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()

        output = (x - mean) / std
        return output, mean, std

    def forward(self, x):

        if self.lf:
            x, mean, std = self.PONO(x)
            mean = self.mean_conv3(self.mean_conv2(self.mean_conv1(mean)))
            std = self.std_conv3(self.std_conv2(self.std_conv1(std)))

        y = self.RB(x)

        y = self.AM(y)
        if self.lf:
            return y * std + mean
        return y


"""
    Guidance-based Upsampling
"""
class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r
    def diff_x(self, input, r):
        assert input.dim() == 4

        left = input[:, :, r:2 * r + 1]
        middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
        right = input[:, :, -1:] - input[:, :, -2 * r - 1:    -r - 1]

        output = torch.cat([left, middle, right], dim=2)

        return output
    def diff_y(self, input, r):
        assert input.dim() == 4

        left = input[:, :, :, r:2 * r + 1]
        middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
        right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:    -r - 1]

        output = torch.cat([left, middle, right], dim=3)

        return output

    def forward(self, x):
        assert x.dim() == 4
        return self.diff_y(self.diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)






# 输出维度为

