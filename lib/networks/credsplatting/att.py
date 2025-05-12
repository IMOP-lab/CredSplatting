from __future__ import annotations
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim1, dim2, N, num_heads, focusing_factor=3):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.N = N
        self.num_heads = num_heads
        head_dim = dim2 // num_heads

        self.Act = nn.ReLU()
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, N, dim2)))
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim2)))
        self.focusing_factor = focusing_factor
        
        self.conv = nn.Conv2d(in_channels=dim1, out_channels=dim2, kernel_size=1, stride=1, padding=0, bias=False)
        # self.one_conv = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=1, stride=1, padding=0, bias=False)
        # self.final_conv = nn.Conv2d(in_channels=dim2, out_channels=dim1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2):

        B, N, _ = input1.shape

        input_q = input1.permute(0,2,1).unsqueeze(-1)
        input_q = self.conv(input_q).permute(0,2,1,3).squeeze(-1)     
        
        q = self.Act(input_q + self.positional_encoding)
        k = self.Act(input2 + self.positional_encoding)
        scale = nn.Softplus()(self.scale)#通道缩放
        q = q / scale
        k = k / scale
        # print('q,k:',q.shape, k.shape)
        
        q_norm = q.norm(dim=-1, keepdim=True)#计算L2范数
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** self.focusing_factor
        k = k ** self.focusing_factor
        q = (q_norm / q.norm(dim=-1, keepdim=True)) * q
        k = (k_norm / k.norm(dim=-1, keepdim=True)) * k
        # print('q,k',q.shape, k.shape)

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3) # (B, heads, N, ,C/heads)
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = input2.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        # print('q,k,v',q.shape, k.shape, v.shape)

        kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v * (N ** -0.5))    # (B, heads, C/heads, C/heads)
        x = q @ kv                                                      # (B, heads, W*H*D,   C/heads)
        # print('x1',x.shape)

        x = x.transpose(2,3).reshape(B, -1, N).permute(0,2,1)
        # v = v.reshape(B * self.num_heads, N, -1).permute(0, 2, 1).contiguous()
        # x = x.unsqueeze(-1) + self.one_conv(v.unsqueeze(-1)).reshape(B, -1, N, 1).contiguous()
        # x = self.final_conv(x).squeeze(-1).permute(0,2,1)
        attn_output = self.sigmoid(x) * input1
        # print(attn_output.shape)

        return attn_output

    
# input1 = torch.randn(4, 20480, 24).cuda() # (B,N,C)
# input2 = torch.randn(4, 20480, 120).cuda()
# att = Attention(dim1=24, dim2=120, N=20480, num_heads=8).cuda()
# output = att(input1,input2)
# print(output.shape)