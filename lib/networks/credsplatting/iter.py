import torch
import torch.nn as nn
from torch.nn import functional as F


# 简单的卷积层+BN+ReLU
def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# 空间注意力模块
class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        # 使用平均池化和最大池化通道信息，生成一个注意力图
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化和最大池化沿通道维度进行特征融合
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(x_out))  # 得到空间注意力图
        # print(attention_map.shape, x.shape, x_out.shape)
        return x * attention_map  # 将输入与注意力图相乘，进行加权

# 轻量级卷积神经网络
class SimpleCNNWithAttention(nn.Module):
    def __init__(self, input_channels=3):
        super(SimpleCNNWithAttention, self).__init__()
        
        # 第一部分：卷积层
        self.layer1 = conv_block(input_channels, 16)
        self.layer2 = conv_block(16, 16)
        
        # 空间注意力模块：插入在第二层卷积后
        self.spatial_attention = SpatialAttentionModule(kernel_size=3)
        
        # 第三部分：卷积层
        self.layer3 = conv_block(16, 1)
        self.sig = nn.Sigmoid()

    
    def forward(self, x):
        # 前向传播
        x = self.layer1(x)
        x = self.layer2(x)
        
        # 加入空间注意力模块
        x = self.spatial_attention(x)
        
        x = self.layer3(x)

        out = self.sig(x)
        
        return out