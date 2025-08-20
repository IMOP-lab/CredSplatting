import torch.nn as nn
from .utils import *

class CostRegNet(nn.Module):
    def __init__(self, in_channels, base = 8, norm_act=nn.BatchNorm3d):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, base, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(base, base*2, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(base*2, base*2, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(base*2, base*4, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(base*4, base*4, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(base*4, base*base, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(base*base, base*base, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base*base, base*4, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(base*4))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base*4, base*2, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(base*2))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base*2, base, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(base))
        self.depth_conv = nn.Sequential(nn.Conv3d(base, 1, 3, padding=1, bias=False))
        self.feat_conv = nn.Sequential(nn.Conv3d(base, base, 3, padding=1, bias=False))

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        feat = self.feat_conv(x)
        depth = self.depth_conv(x)
        return feat, depth.squeeze(1)


class MinCostRegNet(nn.Module):
    def __init__(self, in_channels, base = 8, norm_act=nn.BatchNorm3d):
        super(MinCostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, base, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(base, base*2, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(base*2, base*2, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(base*2, base*4, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(base*4, base*4, norm_act=norm_act)

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base*4, base*2, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(base*2))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base*2, base, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(base))

        self.depth_conv = nn.Sequential(nn.Conv3d(base, 1, 3, padding=1, bias=False))
        self.feat_conv = nn.Sequential(nn.Conv3d(base, base, 3, padding=1, bias=False))

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = conv4
        x = conv2 + self.conv9(conv4)
        # del conv2
        x = conv0 + self.conv11(x)
        # del conv0
        feat = self.feat_conv(x)
        depth = self.depth_conv(x)
        return feat, depth.squeeze(1)



class pred_uncerten(nn.Module):
    def __init__(self, in_channels, base = 8, norm_act=nn.BatchNorm3d):
        super().__init__()
        self.c0 = ConvBnReLU3D(in_channels, base, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(base, base, stride=2, norm_act=norm_act)
        # # self.conv2 = ConvBnReLU3D(base, base, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(base, base, stride=2, norm_act=norm_act)
        # # self.conv4 = ConvBnReLU3D(base*4, base*4, norm_act=norm_act)

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base, base, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(base))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base, base, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(base))
        self.final_conv = nn.Sequential(nn.Conv3d(base, 1, 3, padding=1, bias=False))

        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        conv0 = self.c0(x)
        conv2 = self.conv1(conv0)
        conv4 = self.conv3(conv2)
        x1 = conv2 + self.conv9(conv4)
        x1 = conv0 + self.conv11(x1)
        # print(conv0.shape,self.conv11(conv4).shape)
        max_result,_= torch.max(self.final_conv(x1), dim=2,keepdim=True)
        x = max_result * x

        return x

class SigCostRegNet(nn.Module):
    def __init__(self, in_channels, norm_act=nn.BatchNorm3d):
        super(SigCostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=(1,2,2), pad=(1,1,1),norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=(1,2,2), pad=(1,1,1), norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)
        
        self.conv5 = ConvBnReLU3D(32, 64, stride=(1,2,2), pad=(1,1,1), norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=(1,1,1), output_padding=(0,1,1),
                               stride=(1,2,2), bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=(1,1,1), output_padding=(0,1,1),
                               stride=(1,2,2), bias=False),
            norm_act(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=(1,1,1), output_padding=(0,1,1),
                               stride=(1,2,2), bias=False),
            norm_act(8))

        self.feat_conv = nn.Sequential(nn.Conv3d(8, in_channels, 3, padding=1, bias=False))

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        feat = self.feat_conv(x)
        return feat