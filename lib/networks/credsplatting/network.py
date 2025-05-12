import torch
import torch.nn as nn
from torch.nn import functional as F
from .feature_net import FeatureNet, AutoEncoder
from .cost_reg_net import CostRegNet, MinCostRegNet
from . import utils
from lib.config import cfg
from .gs import GS, GS1, GS1_fuse, GS_agg1
from lib.gaussian_renderer import render
import os
import imageio
import numpy as np
import PIL
import cv2
from .utils import write_cam, save_pfm, visualize_depth
import sys
from .iter import SpatialAttentionModule
from einops import rearrange
import torch.distributed as dist


def convgnrelu(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True, group_channel=8):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class InterViewAAModule(nn.Module):
    def __init__(self,in_channels=32, bias=True):
        super(InterViewAAModule, self).__init__()
        self.reweight_network = nn.Sequential(
                                    convgnrelu(in_channels, 4, kernel_size=3, stride=1, dilation=1, bias=bias),
                                    resnet_block_gn(4, kernel_size=1),
                                    nn.Conv2d(4, 1, kernel_size=1, padding=0),
                                    nn.Sigmoid()
                                )
    
    def forward(self, x):
        return self.reweight_network(x)


def resnet_block_gn(in_channels,  kernel_size=3, dilation=[1,1], bias=True, group_channel=8):
    return ResnetBlockGn(in_channels, kernel_size, dilation, bias=bias, group_channel=group_channel)

class ResnetBlockGn(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias, group_channel=8):
        super(ResnetBlockGn, self).__init__()
        self.stem = nn.Sequential(
            convgnrelu(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], bias=bias, group_channel=group_channel), 
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
            nn.BatchNorm2d(in_channels),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.stem(x) + x
        out = self.relu(out)
        return out

class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        self.feature_net = FeatureNet()

        self.cost_reg_0 = MinCostRegNet(int(32 * (2**(-0))*1))
        self.cost_reg_1 = CostRegNet(int(32 * (2**(-1)*2)*1.5), base=8)
        self.cost_reg_pair = CostRegNet(int(32 * (2**(-1)*2)))

        self.gs_0 = GS(feat_ch=cfg.credsplatting.cas_config.gs_model_feat_ch[0]+3)
        self.gs_1 = GS1(feat_ch=cfg.credsplatting.cas_config.gs_model_feat_ch[1]+3)
        self.gs_1_fuse = GS1_fuse(feat_ch=cfg.credsplatting.cas_config.gs_model_feat_ch[1]+3)
        
        self.conv_transmit = AutoEncoder(54+24+3, 16, 16)
        self.omega = InterViewAAModule(16)

    def render_rays(self, rays, **kwargs):
        level, batch, im_feat, feat_volume, gs_model, size = kwargs['level'], kwargs['batch'], kwargs['im_feat'], kwargs['feature_volume'], kwargs['gs_model'], kwargs['size']
        world_xyz, uvd, z_vals = utils.sample_along_depth(rays, N_samples=cfg.credsplatting.cas_config.num_samples[level], level=level)
        B, N_rays, N_samples = world_xyz.shape[:3]
        rgbs = utils.unpreprocess(batch['src_inps'], render_scale=cfg.credsplatting.cas_config.render_scale[level])
        up_feat_scale = cfg.credsplatting.cas_config.render_scale[level] / cfg.credsplatting.cas_config.im_ibr_scale[level]
        if up_feat_scale != 1.:
            B, S, C, H, W = im_feat.shape
            im_feat = F.interpolate(im_feat.reshape(B*S, C, H, W), None, scale_factor=up_feat_scale, align_corners=True, mode='bilinear').view(B, S, C, int(H*up_feat_scale), int(W*up_feat_scale))

        img_feat_rgb = torch.cat((im_feat, rgbs), dim=2)
        H_O, W_O = kwargs['batch']['src_inps'].shape[-2:]
        B, H, W = len(uvd), int(H_O * cfg.credsplatting.cas_config.render_scale[level]), int(W_O * cfg.credsplatting.cas_config.render_scale[level])
        uvd[..., 0], uvd[..., 1] = (uvd[..., 0]) / (W-1), (uvd[..., 1]) / (H-1)
        
        vox_feat = utils.get_vox_feat(uvd.reshape(B, -1, 3), feat_volume)
        # print(feat_volume.shape, vox_feat.shape)
        img_feat_rgb_dir = utils.get_img_feat(world_xyz, img_feat_rgb, batch, self.training, level)
        
        net_output = gs_model(vox_feat, img_feat_rgb_dir, z_vals, batch, size, level)
        return net_output


    def batchify_rays(self, rays, **kwargs):
        ret = self.render_rays(rays, **kwargs)
        return ret
    
    def batchify_rays_fuse(self, rays, **kwargs):
        ret = self.render_rays_fuse(rays, **kwargs)
        return ret

    
    def render_rays_fuse(self, rays, **kwargs):
        # im_feat 所有源影像特征， feat_volume 当前视角代价体特征
        level, batch, im_feat, feat_volume, gs_model, size, view_flag, prob = kwargs['level'], kwargs['batch'], kwargs['im_feat'], kwargs['feature_volume'], kwargs['gs_model'], kwargs['size'],kwargs['view_flag'],kwargs['prob']
        world_xyz, uvd, z_vals = utils.sample_along_depth(rays, N_samples=cfg.credsplatting.cas_config.num_samples[level], level=level)
        B, N_rays, N_samples = world_xyz.shape[:3]  # [1, 327680, 1]
        rgbs = utils.unpreprocess(batch['src_inps'], render_scale=cfg.credsplatting.cas_config.render_scale[level])
        up_feat_scale = cfg.credsplatting.cas_config.render_scale[level] / cfg.credsplatting.cas_config.im_ibr_scale[level]
        if up_feat_scale != 1.:
            B, S, C, H, W = im_feat.shape
            im_feat = F.interpolate(im_feat.reshape(B*S, C, H, W), None, scale_factor=up_feat_scale, align_corners=True, mode='bilinear').view(B, S, C, int(H*up_feat_scale), int(W*up_feat_scale))

        # print(batch['src_inps'].shape)
        _, mv, _, _, _ = batch['src_inps'].shape
        
        img_feat_rgb = torch.cat((im_feat, rgbs), dim=2)
        # print(img_feat_rgb.shape)
        # print(img_feat_rgb.shape)
        H_O, W_O = kwargs['batch']['src_inps'].shape[-2:]
        B, H, W = len(uvd), int(H_O * cfg.credsplatting.cas_config.render_scale[level]), int(W_O * cfg.credsplatting.cas_config.render_scale[level])
        uvd[..., 0], uvd[..., 1] = (uvd[..., 0]) / (W-1), (uvd[..., 1]) / (H-1)
        vox_feat = utils.get_vox_feat(uvd.reshape(B, -1, 3), feat_volume)
        img_feat_rgb_dir = utils.get_img_feat(world_xyz, img_feat_rgb, batch, self.training, level)

        net_output = gs_model(vox_feat, img_feat_rgb_dir, size)

        # mask
        prob =  F.interpolate(prob, scale_factor=2, align_corners=True, mode='bilinear').squeeze(1)
        mask = prob.reshape(B, N_rays, 1) # version 3 4-2

        # with torch.no_grad():
        #     inv_scale = torch.tensor([W-1, H-1], dtype=torch.float32, device=net_output.device).unsqueeze(0).expand(B, -1)
        #     mask = utils.mask_viewport(world_xyz, kwargs['batch']['src_exts'], kwargs['batch']['src_ixts'], inv_scale)
        #     mask = mask.reshape(B, -1, N_samples)
        #     # mask = torch.ones_like(mask)

        return net_output, mask, z_vals
    
    
    def merge_gs_outputs(self, batch, rets, K, size, level):
        # output: radiance, sigma, world_xyz, rot_out, scale_out, opacity_out, color_out, rgb_vr

        # 融合：每张影像产生一堆参数。对参数加权而不是对渲染的结果图加权
        z_vals = torch.stack([rets[f'zval_view{i}'] for i in range(K)], dim=1) # [B, K, 327680, 1]
        masks = torch.stack([rets[f'mask_view{i}'] for i in range(K)], dim=1)  # [B, K, 327680, 1]
        masks_sum = masks.unsqueeze(1).sum(2)                                  # [B, 1, 327680, 1]
        # print(torch.unique(masks_sum))
        masks = torch.where(masks_sum > 0, masks / masks_sum, 1 / K)
        # print(torch.unique(masks))

        radiances = []
        sigmas = []
        nerf_feats = []

        for i in range(K):
            radiance_i, sigma_i, feature_nerf_i = rets[f'output_view{i}']
            radiances.append(radiance_i)
            sigmas.append(sigma_i)
            nerf_feats.append(feature_nerf_i)

        radiances = torch.stack(radiances, dim=1)         # [B, K, 327680, 3]
        sigmas = torch.stack(sigmas, dim=1)               # [B, K, 327680, 1] 
        nerf_feats = torch.stack(nerf_feats, dim=1)       # [B, K, 327680, 24]

        net_output = self.gs_1_fuse(sigmas, radiances, nerf_feats, masks, z_vals, batch, size, level)   # net_output: world_xyz, rot_out, scale_out, opacity_out, color_out, rgb_vr_full, depth_vr_full

        return net_output



    def forward_feat(self, x):
        # print(x.shape,f"Current rank: {dist.get_rank()}")
        B, S, C, H, W = x.shape
        x = x.view(B*S, C, H, W)
        feat2, feat1, feat0 = self.feature_net(x)
        feats = {
                'level_2': feat0.reshape((B, S, feat0.shape[1], H, W)),
                'level_1': feat1.reshape((B, S, feat1.shape[1], H//2, W//2)),
                'level_0': feat2.reshape((B, S, feat2.shape[1], H//4, W//4)),
                }
        return feats

    def forward_render(self, ret, batch):
        B, _, _, H, W = batch['src_inps'].shape
        rgb = ret['rgb'].reshape(B, H, W, 3).permute(0, 3, 1, 2)
        rgb = self.cnn_renderer(rgb)
        ret['rgb'] = rgb.permute(0, 2, 3, 1).reshape(B, H*W, 3)


    def forward(self, batch):
        B, _, _, H_img, W_img = batch['src_inps'].shape
        _, _, _, H, W = batch['src_inps'].shape
        if not cfg.save_video:
            feats = self.forward_feat(batch['src_inps'])
            ret = {}
            #=========================== Stage_0 init ====================================
            depth0, std0, near_far0 = None, None, None
            tgt_fea = None
            # print(12455533)
            H0, W0 = int(H_img*cfg.credsplatting.cas_config.render_scale[0]), int(W_img*cfg.credsplatting.cas_config.render_scale[0])
            feature_volume0, depth_values0, near_far0, paired_feature_volume, paired_reweight = utils.build_feature_volume3(
                    feats['level_0'],
                    batch,
                    D=cfg.credsplatting.cas_config.volume_planes[0],
                    depth=depth0,
                    std=std0,
                    near_far = near_far0,
                    level=0,)
            feature_volume0, depth_prob0 = self.cost_reg_0(feature_volume0)
            depth0, std0 = utils.depth_regression(depth_prob0, depth_values0, 0, batch)
            ret_0 = {}
            rays0 = utils.build_rays(depth0, std0, batch, self.training, near_far0, 0)
            im_feat_level = cfg.credsplatting.cas_config.render_im_feat_level[0]
            output0 = self.batchify_rays(
                    rays=rays0,
                    feature_volume=feature_volume0,
                    batch=batch,
                    im_feat=feats[f'level_{im_feat_level}'],
                    gs_model=self.gs_0,
                    level=0,
                    size=(H0, W0))
            world_xyz, rot_out, scale_out, opacity_out, color_out, rgb_vr0, depth2, fea_transmit = output0
            gs_novel0 = []
            render_novel0 = []
            for b_i in range(B):
                render_novel_i_0  = render(batch['novel_view0'], b_i, world_xyz[b_i], color_out[b_i], rot_out[b_i], scale_out[b_i], opacity_out[b_i], bg_color=cfg.credsplatting.bg_color)
                if cfg.credsplatting.reweighting:
                        render_novel_i = (render_novel_i_0 + rgb_vr0[b_i] * 4) / 5
                else:
                    render_novel_i = (render_novel_i_0 + rgb_vr0[b_i]) / 2
                gs_novel0.append(render_novel_i_0)
                render_novel0.append(render_novel_i)

            gs_novel0 = torch.stack(gs_novel0)
            render_novel0 = torch.stack(render_novel0)

            ret_0.update({'rgb': render_novel0.flatten(2).permute(0,2,1)})
            if cfg.credsplatting.cas_config.depth_inv[0]:
                ret_0.update({'depth_mvs': 1./ depth0})
            else:
                ret_0.update({'depth_mvs': depth0})
            ret_0.update({'std': std0})
            if ret_0['rgb'].isnan().any():
                __import__('ipdb').set_trace()
            ret.update({key+'_level0': ret_0[key] for key in ret_0})

            #=========================== Stage_1 ====================================
            fea_transmit = fea_transmit.reshape(B,H0, W0, fea_transmit.shape[-1]).permute(0,3,1,2)
            fea_transmit = torch.cat((fea_transmit, gs_novel0, rgb_vr0, render_novel0), dim=1)
            _, _, h, w = fea_transmit.shape
            tgt_fea = self.conv_transmit(fea_transmit)
            

            src_feats = feats['level_1']   # [f'level_{1}']  # [16 , 1/2]
            H1, W1 = int(H_img*cfg.credsplatting.cas_config.render_scale[1]), int(W_img*cfg.credsplatting.cas_config.render_scale[1])
            feature_volume1, depth_values1, near_far1, paired_feature_volume, paired_reweight = utils.build_feature_volume3(
                    src_feats,
                    batch,
                    D=cfg.credsplatting.cas_config.volume_planes[1],
                    depth=depth0,
                    std=std0,
                    near_far = near_far0,
                    level=1,
                    tgt_fea=tgt_fea,
                    omega_net = self.omega,)
            # print(11111112334)
            feature_volume1, depth_prob1 = self.cost_reg_1(feature_volume1)
            depth_fuse, std_fuse = utils.depth_regression(depth_prob1, depth_values1, 1, batch)

            
            # paired 
            ret_1 = {}
            re = {}
            k = 0
            depth_fuse_S = []
            
            _,S,_,_,_ = batch['src_inps'].shape
            paired_reweight2 = torch.cat(paired_reweight, dim=1)
            paired_reweight2 = rearrange(paired_reweight2, 'b s c h w -> (b s) c h w')
            paired_reweight2 = F.interpolate(paired_reweight2, None, scale_factor=2, align_corners=True, mode='bilinear')
            paired_reweight2 = rearrange(paired_reweight2, '(b s) c h w -> b s c h w', b=B, s=S)
            
            # print(paired_reweight2.shape)
            for fvp, reweight in zip(paired_feature_volume, paired_reweight):
                paired_fvp, prob_volume = self.cost_reg_pair(fvp)
                depth_fuse_1, _ = utils.depth_regression(prob_volume, depth_values1, 1, batch)
                reweight = F.softmax(reweight.squeeze(1), 1) # [b, d, h, w]
                prob, _ = reweight.max(1)
                prob = prob.unsqueeze(1)

                rays_1 = utils.build_rays(depth_fuse, std_fuse, batch, self.training, near_far1, 1)
                im_feat_level = cfg.credsplatting.cas_config.render_im_feat_level[1]       # 1/4
                output_1, mask_1, z_val_1 = self.batchify_rays_fuse(
                        rays=rays_1,
                        feature_volume=paired_fvp,
                        batch=batch,
                        im_feat=feats[f'level_{im_feat_level}'] * paired_reweight2,
                        gs_model=self.gs_1,
                        view_flag=k,
                        prob=prob,
                        level=1,
                        size=(H1, W1))
                # output: radiance, sigma, world_xyz, rot_out, scale_out, opacity_out, color_out, rgb_vr
                re.update({f'output_view{k}': output_1, f'mask_view{k}':mask_1, f'zval_view{k}':z_val_1})
                k = k + 1
                depth_fuse_S.append(depth_fuse_1)
                # if k==1:
                #     break

                
            net_output = self.merge_gs_outputs(batch, re, k, size=(H1, W1), level=1)
            world_xyz_fused, rot_out_fused, scale_out_fused, opacity_out_fused, color_out_fused, rgb_vr_full, depth_vr_full = net_output

            depth_fuse_S = torch.stack(depth_fuse_S, dim=1)

            # fuse gs rendering
            render_novel_1 = []
            gs_novel_1 = []
            for b_i in range(B):
                render_outputs  = render(batch['novel_view1'], b_i, world_xyz_fused[b_i], color_out_fused[b_i], rot_out_fused[b_i], scale_out_fused[b_i], opacity_out_fused[b_i], bg_color=cfg.credsplatting.bg_color)
                render_novel_i_1 = render_outputs   # [3, H, W]
                
                if cfg.credsplatting.reweighting: 
                    render_novel_i = (render_novel_i_1 + rgb_vr_full[b_i] * 4) / 5
                else:
                    render_novel_i = (render_novel_i_1 + rgb_vr_full[b_i]) / 2

                render_novel_1.append(render_novel_i)
                gs_novel_1.append(render_novel_i_1)

            render_novel_1 = torch.stack(render_novel_1)   # 1
            gs_novel_1 = torch.stack(gs_novel_1)
            
            # print(render_novel_1.shape)
            ret_1.update({'rgb': render_novel_1.flatten(2).permute(0, 2, 1)})
            if cfg.credsplatting.cas_config.depth_inv[1]:
                ret_1.update({'depth_mvs': 1./depth_fuse})
            else:
                ret_1.update({'depth_mvs': depth_fuse})
            ret_1.update({'depth_mvs2': depth_vr_full})
            # ret_1.update({'depth_mvs3': dpt})
            ret_1.update({'std': std_fuse})
            if ret_1['rgb'].isnan().any():
                __import__('ipdb').set_trace()

            ret_1.update({'depth_fuse_S': depth_fuse_S})

            ret.update({key+'_level1': ret_1[key] for key in ret_1})

            if cfg.save_ply:
                result_dir = cfg.dir_ply
                #保存gs参数
                for b_i in range(B):
                    scan_dir = os.path.join(result_dir, batch['meta']['scene'][b_i])
                    os.makedirs(scan_dir, exist_ok = True)
                    gs_dir = os.path.join(scan_dir, 'gs_para')
                    os.makedirs(gs_dir, exist_ok = True)
                
                    gs_path = os.path.join(gs_dir, '{}_{}_{}.png'.format(batch['meta']['scene'][b_i], batch['meta']['tar_view'][b_i].item(), batch['meta']['frame_id'][b_i].item())).replace('.png', '.pth')
                    # print(gs_path)
                
                    # sys.exit()
                    # 创建一个字典来存储这些参数
                    data_dict = {
                        'world_xyz_fused': world_xyz_fused[b_i] / cfg.credsplatting.scale_factor,
                        'color_out_fused': color_out_fused[b_i],
                        'rot_out_fused': rot_out_fused[b_i],
                        'scale_out_fused': scale_out_fused[b_i],
                        'opacity_out_fused': opacity_out_fused[b_i]
                    }
                    # 保存字典到文件
                    torch.save(data_dict, gs_path)

                ##保存深度图
                render_novel = render_novel_1
                # print(depth_fuse.shape, torch.unique(depth_fuse))
                # print(render_novel_1.shape, torch.unique(render_novel_1))
                # B, _, _, H, W = batch['src_inps'].shape
                result_dir = cfg.dir_ply
                os.makedirs(result_dir, exist_ok = True)
                # depth = F.interpolate(depth.unsqueeze(1),size=(H,W)).squeeze(1)
                depth = F.interpolate(depth_fuse.unsqueeze(1),size=(H,W)).squeeze(1)
                # print(depth.shape, torch.unique(depth))
                # print(render_novel.shape, torch.unique(render_novel))
                
                for b_i in range(B):
                    scan_dir = os.path.join(result_dir, batch['meta']['scene'][b_i])
                    os.makedirs(scan_dir, exist_ok = True)
                    img_dir = os.path.join(scan_dir, 'images')
                    os.makedirs(img_dir, exist_ok = True)
                    img_path = os.path.join(img_dir, '{}_{}_{}.png'.format(batch['meta']['scene'][b_i], batch['meta']['tar_view'][b_i].item(), batch['meta']['frame_id'][b_i].item()))
                    img = render_novel[b_i].permute(1,2,0).detach().cpu().numpy()
                    img = (img*255).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(img_path, img)
                    cam_dir = os.path.join(scan_dir, 'cam')
                    os.makedirs(cam_dir, exist_ok = True)
                    cam_path = os.path.join(cam_dir, '{}_{}_{}.txt'.format(batch['meta']['scene'][b_i], batch['meta']['tar_view'][b_i].item(), batch['meta']['frame_id'][b_i].item()))
                    ixt = batch['tar_ixt'].detach().cpu().numpy()[b_i]
                    ext = batch['tar_ext'].detach().cpu().numpy()[b_i]
                    # print(cfg.credsplatting.scale_factor)
                    ext[:3,3] /= cfg.credsplatting.scale_factor
                    write_cam(cam_path, ixt, ext)

                    depth /= cfg.credsplatting.scale_factor
                    depth_dir = os.path.join(scan_dir, 'depth')
                    os.makedirs(depth_dir, exist_ok = True)
                    depth_path = os.path.join(depth_dir, '{}_{}_{}.pfm'.format(batch['meta']['scene'][b_i], batch['meta']['tar_view'][b_i].item(), batch['meta']['frame_id'][b_i].item()))
                    depth_vis = depth[b_i].detach().cpu().numpy()
                    save_pfm(depth_path, depth_vis)
                    
                    depth_minmax = [
                        batch["near_far"].min().detach().cpu().numpy()/(1.5*cfg.credsplatting.scale_factor),
                        batch["near_far"].max().detach().cpu().numpy()/ (1.5*cfg.credsplatting.scale_factor),
                    ]
                    rendered_depth_vis, _ = visualize_depth(depth_vis, depth_minmax)
                    rendered_depth_vis = rendered_depth_vis.permute(1,2,0).detach().cpu().numpy()
                    depth_vis_path = os.path.join(depth_dir, '{}_{}_{}.png'.format(batch['meta']['scene'][b_i], batch['meta']['tar_view'][b_i].item(), batch['meta']['frame_id'][b_i].item()))
                    imageio.imwrite(depth_vis_path, (rendered_depth_vis*255.).astype(np.uint8))
            # sys.exit()

            return ret
        else:
            pred_rgb_nb_list = []
            for v_i, (meta0, meta) in enumerate(zip(batch['rendering_video_meta0'], batch['rendering_video_meta1'])):
                batch['tar_ext'][:,:3] = meta['tar_ext'][:,:3]
                batch['rays_0'] = meta['rays_0']
                batch['rays_1'] = meta['rays_1']
                feats = self.forward_feat(batch['src_inps'])
                depth0, std0, near_far0 = None, None, None
                tgt_fea = None

                H0, W0 = int(H_img*cfg.credsplatting.cas_config.render_scale[0]), int(W_img*cfg.credsplatting.cas_config.render_scale[0])
                feature_volume0, depth_values0, near_far0, paired_feature_volume, paired_reweight = utils.build_feature_volume3(
                        feats['level_0'],
                        batch,
                        D=cfg.credsplatting.cas_config.volume_planes[0],
                        depth=depth0,
                        std=std0,
                        near_far = near_far0,
                        level=0,)
                feature_volume0, depth_prob0 = self.cost_reg_0(feature_volume0)
                depth0, std0 = utils.depth_regression(depth_prob0, depth_values0, 0, batch)
                ret_0 = {}
                rays0 = utils.build_rays(depth0, std0, batch, self.training, near_far0, 0)
                im_feat_level = cfg.credsplatting.cas_config.render_im_feat_level[0]
                output0 = self.batchify_rays(
                        rays=rays0,
                        feature_volume=feature_volume0,
                        batch=batch,
                        im_feat=feats[f'level_{im_feat_level}'],
                        gs_model=self.gs_0,
                        level=0,
                        size=(H0, W0))
                world_xyz, rot_out, scale_out, opacity_out, color_out, rgb_vr0, depth2, fea_transmit = output0
                gs_novel0 = []
                render_novel0 = []
                for b_i in range(B):
                    render_novel_i_0  = render(meta0, b_i, world_xyz[b_i], color_out[b_i], rot_out[b_i], scale_out[b_i], opacity_out[b_i], bg_color=cfg.credsplatting.bg_color)
                    if cfg.credsplatting.reweighting:
                            render_novel_i = (render_novel_i_0 + rgb_vr0[b_i] * 4) / 5
                    else:
                        render_novel_i = (render_novel_i_0 + rgb_vr0[b_i]) / 2
                    gs_novel0.append(render_novel_i_0)
                    render_novel0.append(render_novel_i)

                gs_novel0 = torch.stack(gs_novel0)
                render_novel0 = torch.stack(render_novel0)

                fea_transmit = fea_transmit.reshape(B,H0, W0, fea_transmit.shape[-1]).permute(0,3,1,2)
                fea_transmit = torch.cat((fea_transmit, gs_novel0, rgb_vr0, render_novel0), dim=1)
                _, _, h, w = fea_transmit.shape
                tgt_fea = self.conv_transmit(fea_transmit)

                src_feats = feats['level_1']   # [f'level_{1}']  # [16 , 1/2]
                H1, W1 = int(H_img*cfg.credsplatting.cas_config.render_scale[1]), int(W_img*cfg.credsplatting.cas_config.render_scale[1])
                feature_volume1, depth_values1, near_far1, paired_feature_volume, paired_reweight = utils.build_feature_volume3(
                        src_feats,
                        batch,
                        D=cfg.credsplatting.cas_config.volume_planes[1],
                        depth=depth0,
                        std=std0,
                        near_far = near_far0,
                        level=1,
                        tgt_fea=tgt_fea,
                        omega_net = self.omega,)
                
                feature_volume1, depth_prob1 = self.cost_reg_1(feature_volume1)
                depth_fuse, std_fuse = utils.depth_regression(depth_prob1, depth_values1, 1, batch)

                # paired 
                ret_1 = {}
                re = {}
                k = 0
                depth_fuse_S = []
                
                _,S,_,_,_ = batch['src_inps'].shape
                paired_reweight2 = torch.cat(paired_reweight, dim=1)
                paired_reweight2 = rearrange(paired_reweight2, 'b s c h w -> (b s) c h w')
                paired_reweight2 = F.interpolate(paired_reweight2, None, scale_factor=2, align_corners=True, mode='bilinear')
                paired_reweight2 = rearrange(paired_reweight2, '(b s) c h w -> b s c h w', b=B, s=S)
                
                # print(paired_reweight2.shape)
                for fvp, reweight in zip(paired_feature_volume, paired_reweight):
                    paired_fvp, prob_volume = self.cost_reg_pair(fvp)
                    depth_fuse_1, _ = utils.depth_regression(prob_volume, depth_values1, 1, batch)
                    reweight = F.softmax(reweight.squeeze(1), 1) # [b, d, h, w]
                    prob, _ = reweight.max(1)
                    prob = prob.unsqueeze(1)

                    rays_1 = utils.build_rays(depth_fuse, std_fuse, batch, self.training, near_far1, 1)
                    im_feat_level = cfg.credsplatting.cas_config.render_im_feat_level[1]       # 1/4

                    output_1, mask_1, z_val_1 = self.batchify_rays_fuse(
                            rays=rays_1,
                            feature_volume=paired_fvp,
                            batch=batch,
                            im_feat=feats[f'level_{im_feat_level}'] * paired_reweight2,
                            gs_model=self.gs_1,
                            view_flag=k,
                            prob=prob,
                            level=1,
                            size=(H1, W1))
                    re.update({f'output_view{k}': output_1, f'mask_view{k}':mask_1, f'zval_view{k}':z_val_1})
                    k = k + 1
                    depth_fuse_S.append(depth_fuse_1)

                    
                net_output = self.merge_gs_outputs(batch, re, k, size=(H1, W1), level=1)
                world_xyz_fused, rot_out_fused, scale_out_fused, opacity_out_fused, color_out_fused, rgb_vr_full, depth_vr_full = net_output

                depth_fuse_S = torch.stack(depth_fuse_S, dim=1)
                # fuse gs rendering
                render_novel_1 = []
                gs_novel_1 = []
                for b_i in range(B):
                    render_outputs  = render(meta, b_i, world_xyz_fused[b_i], color_out_fused[b_i], rot_out_fused[b_i], scale_out_fused[b_i], opacity_out_fused[b_i], bg_color=cfg.credsplatting.bg_color)
                    
                    render_novel_i_1 = render_outputs   # [3, H, W]
                    if cfg.credsplatting.reweighting: 
                        render_novel_i = (render_novel_i_1 + rgb_vr_full[b_i] * 4) / 5
                    else:
                        render_novel_i = (render_novel_i_1 + rgb_vr_full[b_i]) / 2
                    
                    render_novel_i = render_novel_i.permute(1,2,0)
                    if cfg.credsplatting.eval_center:
                        H_crop, W_crop = int(H_img*0.1), int(W_img*0.1)
                        render_novel_i = render_novel_i[H_crop:-H_crop, W_crop:-W_crop,:]
                        
                    if v_i == 0:
                        pred_rgb_nb_list.append([(render_novel_i.data.cpu().numpy()*255).astype(np.uint8)])
                    else:
                        pred_rgb_nb_list[b_i].append((render_novel_i.data.cpu().numpy()*255).astype(np.uint8))
                    img_dir = os.path.join(cfg.result_dir, '{}_{}'.format(batch['meta']['scene'][b_i], batch['meta']['tar_view'][b_i].item()))
                    os.makedirs(img_dir,exist_ok=True)
                    save_path = os.path.join(img_dir,f'{len(pred_rgb_nb_list[b_i])}.png')
                    print(save_path)
                    PIL.Image.fromarray((render_novel_i.data.cpu().numpy()*255).astype(np.uint8)).save(save_path)

            for b_i in range(B):
                video_path = os.path.join(cfg.result_dir, '{}_{}_{}.mp4'.format(batch['meta']['scene'][b_i], batch['meta']['tar_view'][b_i].item(), batch['meta']['frame_id'][b_i].item()))
                imageio.mimwrite(video_path, np.stack(pred_rgb_nb_list[b_i]), fps=10, quality=10)