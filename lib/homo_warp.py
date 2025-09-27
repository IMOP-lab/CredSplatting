import torch.nn as nn
import torch
from kornia.utils import create_meshgrid
import torch.nn.functional as F
import math
import cv2
import sys
import torchvision.transforms as T
from PIL import Image
import numpy as np

def get_proj_mats(src_inps, src_ext, src_ixt, tar_ext, tar_ixt,src_scale, tar_scale):

    #加上batch维度以适配代码
    src_inps = src_inps.unsqueeze(0)
    src_ext = src_ext.unsqueeze(0)
    src_ixt = src_ixt.unsqueeze(0)
    tar_ext = tar_ext.unsqueeze(0)
    tar_ixt = tar_ixt.unsqueeze(0)

    # print(src_inps.shape, src_ext.shape, src_ixt.shape, tar_ext.shape, tar_ixt.shape)

    B, S_V, C, H, W = src_inps.shape
    # src_ext = src_exts
    # src_ixt = src_ixts.clone()
    src_ixt[:, :, :2] *= src_scale
    src_projs = src_ixt @ src_ext[:, :, :3]
    # tar_ext = batch['tar_ext']
    # tar_ixt = batch['tar_ixt'].clone()
    tar_ixt[:, :2] *= tar_scale
    tar_projs = tar_ixt @ tar_ext[:, :3]
    tar_ones = torch.zeros((B, 1, 4)).to(tar_projs.device)
    tar_ones[:, :, 3] = 1
    tar_projs = torch.cat((tar_projs, tar_ones), dim=1)
    tar_projs_inv = torch.inverse(tar_projs)

    src_projs = src_projs.view(B, S_V, 3, 4)
    tar_projs_inv = tar_projs_inv.view(B, 1, 4, 4)

    proj_mats = src_projs @ tar_projs_inv
    return proj_mats


def get_proj_mats2(src_inps, src_ext, src_ixt, tar_ext, tar_ixt,src_scale, tar_scale):

    #加上batch维度以适配代码
    src_inps = src_inps.unsqueeze(1)
    src_ext = src_ext.unsqueeze(1)
    src_ixt = src_ixt.unsqueeze(1)
    tar_ext = tar_ext.unsqueeze(0)
    tar_ixt = tar_ixt.unsqueeze(0)

    print(src_inps.shape, src_ext.shape, src_ixt.shape, tar_ext.shape, tar_ixt.shape)

    B, S_V, C, H, W = src_inps.shape
    tar_ext = tar_ext.repeat(B, 1, 1)
    tar_ixt = tar_ixt.repeat(B, 1, 1)

    # src_ext = src_exts
    # src_ixt = src_ixts.clone()
    src_ixt[:, :, :2] *= src_scale
    src_projs = src_ixt @ src_ext[:, :, :3]
    # tar_ext = batch['tar_ext']
    # tar_ixt = batch['tar_ixt'].clone()
    tar_ixt[:, :2] *= tar_scale
    tar_projs = tar_ixt @ tar_ext[:, :3]
    tar_ones = torch.zeros((B, 1, 4)).to(tar_projs.device)
    tar_ones[:, :, 3] = 1
    tar_projs = torch.cat((tar_projs, tar_ones), dim=1)
    tar_projs_inv = torch.inverse(tar_projs)

    src_projs = src_projs.view(B, S_V, 3, 4)
    tar_projs_inv = tar_projs_inv.view(B, 1, 4, 4)

    proj_mats = src_projs @ tar_projs_inv
    return proj_mats


def homo_warp(src_feat, proj_mat, depth_values):
    B, D, H_T, W_T = depth_values.shape
    C, H_S, W_S = src_feat.shape[1:]
    device = src_feat.device

    R = proj_mat[:, :, :3] # (B, 3, 3)
    T = proj_mat[:, :, 3:] # (B, 3, 1)
    # create grid from the ref frame
    ref_grid = create_meshgrid(H_T, W_T, normalized_coordinates=False,
                               device=device) # (1, H, W, 2)
    ref_grid = ref_grid.permute(0, 3, 1, 2) # (1, 2, H, W)
    ref_grid = ref_grid.reshape(1, 2, H_T*W_T) # (1, 2, H*W)
    ref_grid = ref_grid.expand(B, -1, -1) # (B, 2, H*W)
    ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:,:1])), 1) # (B, 3, H*W)
    ref_grid_d = ref_grid.repeat(1, 1, D) # (B, 3, D*H*W)
    src_grid_d = R @ ref_grid_d + T/depth_values.view(B, 1, D*H_T*W_T)
    del ref_grid_d, ref_grid, proj_mat, R, T, depth_values # release (GPU) memory

    src_grid = src_grid_d[:, :2] / torch.clamp_min(src_grid_d[:, 2:], 1e-6) # divide by depth (B, 2, D*H*W)
    src_grid[:, 0] = (src_grid[:, 0])/((W_S - 1) / 2) - 1 # scale to -1~1
    src_grid[:, 1] = (src_grid[:, 1])/((H_S - 1) / 2) - 1 # scale to -1~1
    src_grid = src_grid.permute(0, 2, 1) # (B, D*H*W, 2)
    src_grid = src_grid.view(B, D, H_T*W_T, 2)

    warped_src_feat = F.grid_sample(src_feat, src_grid,
                                    mode='bilinear', padding_mode='zeros',
                                    align_corners=True) # (B, C, D, H*W)
    warped_src_feat = warped_src_feat.view(B, C, D, H_T, W_T)
    src_grid = src_grid.view(B, D, H_T, W_T, 2)

    return warped_src_feat, src_grid
