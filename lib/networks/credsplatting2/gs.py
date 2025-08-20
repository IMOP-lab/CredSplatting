import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
from .utils import *
from .feature_net import Unet
from .cost_reg_net import SigCostRegNet

class GS(nn.Module):
    def __init__(self, hid_n=64, feat_ch=16+3):
        super(GS, self).__init__()
        self.hid_n = hid_n
        self.agg = Agg(feat_ch)
        self.head_dim = 24
        self.Unet = Unet(self.head_dim+24, 24)
        self.opacity_head = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 1),
            nn.Sigmoid()
        )
        # self.rotation_head = nn.Sequential(
        #     nn.Linear(self.head_dim, self.head_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.head_dim, 4),
        # )
        self.scale_head = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 3),
            nn.Softplus()
        )
        self.color = nn.Sequential(
            nn.Linear(feat_ch+self.head_dim+4+self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 1),
            nn.ReLU())
        self.sigma = nn.Sequential(nn.Linear(self.head_dim, 1), nn.Softplus())
        self.color_gs = nn.Sequential(
            nn.Linear(self.head_dim+3, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 3),
            nn.Sigmoid()
        )
        self.lr0 = nn.Sequential(nn.Linear(8+16, self.head_dim), nn.ReLU())
        self.lr1 = nn.Sequential(nn.Linear(self.head_dim, self.head_dim), nn.ReLU())

        self.sigma.apply(weights_init)
        self.color.apply(weights_init)
        self.lr0.apply(weights_init)
        self.lr1.apply(weights_init)

        self.regnet = SigCostRegNet(self.head_dim)

        
    def forward(self, vox_feat, img_feat_rgb_dir, z_vals, batch, size, level):
        H,W = size
        B, N_points, N_views = img_feat_rgb_dir.shape[:-1]
        S = img_feat_rgb_dir.shape[2]
        img_feat = self.agg(img_feat_rgb_dir)
        vox_img_feat = torch.cat((vox_feat, img_feat), dim=-1)

        # print(vox_img_feat.shape)
        x = self.lr0(vox_img_feat)
        x = x.reshape(B,H,W,-1,x.shape[-1])
        x = x.permute(0,4,3,1,2)
        x = self.regnet(x)
        # print(x.shape)
        x = x.permute(0,1,3,4,2).flatten(2).permute(0,2,1)

        x = self.lr1(x)
        
        # depth
        d = z_vals.shape[-1]
        z_vals = z_vals.reshape(B,H,W,d)
        if cfg.credsplatting.cas_config.depth_inv[level]:
            z_vals = 1./torch.clamp_min(z_vals, 1e-6) # to disp
        depth = z_vals.permute(0,3,1,2)
        
        # sigma head
        sigma = self.sigma(x)
        # print(x.shape)
        x0 = torch.cat((x, vox_img_feat), dim=-1)
        fea = x0.clone()
        # radiance head
        x0 = x0.unsqueeze(2).repeat(1,1,S,1)
        # x = x.view(B, -1, 1, x.shape[-1]).repeat(1, 1, S, 1)
        x0 = torch.cat((x0, img_feat_rgb_dir), dim=-1)
        color_weight = F.softmax(self.color(x0), dim=-2)
        radiance = torch.sum((img_feat_rgb_dir[..., -7:-4] * color_weight), dim=-2)
        
        # volume rendering branch
        sigma = sigma.reshape(B,H*W,d)
        raw2alpha = lambda raw: 1.-torch.exp(-raw)
        alpha = raw2alpha(sigma)  # [N_rays, N_samples]
        T = torch.cumprod(1.-alpha+1e-10, dim=-1)[..., :-1]
        T = torch.cat([torch.ones_like(alpha[..., 0:1]), T], dim=-1)
        weights = alpha * T
        radiance = radiance.reshape(B,H*W,d,3)
        rgb_vr = torch.sum(weights[...,None] * radiance, -2) 
        rgb_vr = rgb_vr.reshape(B,H,W,3).permute(0,3,1,2)

        # print(weights.shape, fea.shape)
        # fea = fea.reshape(fea.shape[0],-1,weights.shape[-1],fea.shape[-1])
        # fea = torch.sum(weights[...,None] * fea, -2)  # [N_rays, 3]
        

        x = torch.cat((x, vox_img_feat), dim=-1)
        # print(x.shape)
        # enhance features using a UNet
        x = x.reshape(B,H*W,d,x.shape[-1])
        # print(x.shape, weights.shape)
        x = torch.sum(weights[...,None].unsqueeze(0) * x, -2)
        x = x.reshape(B,H,W,x.shape[-1]).permute(0,3,1,2)
        x = self.Unet(x)
        x = x.flatten(-2).permute(0,2,1)
        
        # print(x.shape, x0.shape, vox_feat.shape, img_feat.shape)
        fea_transmit = torch.cat((x, fea), dim=-1)
        # gs branch
        # rot head
        rot_out = torch.ones((B,x.shape[1], 4)).to(x.device)
        # rot_out = self.rotation_head(x)
        rot_out = torch.nn.functional.normalize(rot_out, dim=-1)
      
        # scale head
        scale_out = self.scale_head(x)
        
        # opacity head
        opacity_out = self.opacity_head(x)
        
        # color head
        x0 = torch.cat((x,rgb_vr.flatten(2).permute(0,2,1)),dim=-1)
        color_out = self.color_gs(x0)
        
        # world_xyz
        weights = weights.reshape(B,H,W,d).permute(0,3,1,2)
        depth = torch.sum(weights * depth, 1) # B H W
        # depth = torch.sum(depth, 1) # B H W
        ext = batch['tar_ext']
        ixt = batch['tar_ixt'].clone()
        ixt[:,:2] *= cfg.credsplatting.cas_config.render_scale[level]
        world_xyz = depth2point(depth, ext, ixt)
        
        # print("depth:::", scale_out.shape)

        return world_xyz, rot_out, scale_out, opacity_out, color_out, rgb_vr, depth, fea_transmit

class GS1(nn.Module):
    def __init__(self, hid_n=64, feat_ch=16+3):
        super(GS1, self).__init__()
        self.hid_n = hid_n
        self.agg = Agg(feat_ch)
        self.head_dim = 24

        self.lr0 = nn.Sequential(nn.Linear(8+16, self.head_dim),
                                 nn.ReLU())
        # self.lrs = nn.ModuleList([
        #     nn.Sequential(nn.Linear(hid_n, hid_n), nn.ReLU()) for i in range(0)])
        self.sigma = nn.Sequential(nn.Linear(self.head_dim, 1), nn.Softplus())
        self.color = nn.Sequential(
            nn.Linear(feat_ch+self.head_dim+4+self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 1),
            nn.ReLU())
        
        self.lr0 = nn.Sequential(nn.Linear(8+16, self.head_dim), nn.ReLU())
        self.lr1 = nn.Sequential(nn.Linear(self.head_dim, self.head_dim), nn.ReLU())

        self.sigma.apply(weights_init)
        self.color.apply(weights_init)
        self.lr0.apply(weights_init)
        self.lr1.apply(weights_init)

        self.regnet = SigCostRegNet(self.head_dim)
        
    def forward(self, vox_feat, img_feat_rgb_dir, size):
        H,W = size
        B, N_points, N_views = img_feat_rgb_dir.shape[:-1]
        S = img_feat_rgb_dir.shape[2] # 3
        img_feat = self.agg(img_feat_rgb_dir)
        vox_img_feat = torch.cat((vox_feat, img_feat), dim=-1)  # vox_img_feat [1, 327680, 24]
        
        # feature_nerf = vox_img_feat.clone()

        x = self.lr0(vox_img_feat)
        x = x.reshape(B,H,W,-1,x.shape[-1])
        x = x.permute(0,4,3,1,2)
        x = self.regnet(x)
        # print(x.shape)
        x = x.permute(0,1,3,4,2).flatten(2).permute(0,2,1)

        x = self.lr1(x)

        feature_nerf = torch.cat((x, vox_img_feat), dim=-1)
        # radiance head
        # x = self.lr0(vox_img_feat) # [B, N_rays, 24]
        
        sigma = self.sigma(x) # [B, N_rays, 1]
        # x = torch.cat((x, vox_img_feat), dim=-1)

        x = torch.cat((x, vox_img_feat), dim=-1)

        # x = x.view(B, -1, 1, x.shape[-1]).repeat(1, 1, S, 1)
        x0 = x.unsqueeze(2).repeat(1,1,S,1)   # [1, N_rays, 3, 24]
        x0 = torch.cat((x0, img_feat_rgb_dir), dim=-1) # [1, N_rays, 3, 63]
        color_weight = F.softmax(self.color(x0), dim=-2) # [1, N_rays, 3, 1]
        # print(img_feat_rgb_dir[..., -7:-4].shape)        # [1, N_rays, 3, 3]
        radiance = torch.sum((img_feat_rgb_dir[..., -7:-4] * color_weight), dim=-2) # [1, N_rays, 3]

        return radiance, sigma, feature_nerf


class GS1_fuse(nn.Module):
    def __init__(self, hid_n=64, feat_ch=16+3):
        super(GS1_fuse, self).__init__()
        self.hid_n = hid_n
        self.head_dim = 24
        self.Unet = Unet(self.head_dim, 24)
        self.lr0 = nn.Sequential(nn.Linear(48, self.head_dim),nn.ReLU())

        self.opacity_head = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 1),
            nn.Sigmoid()
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 4),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 3),
            nn.Softplus()
        )
        # self.color = nn.Sequential(
        #     nn.Linear(feat_ch+self.head_dim+4, self.head_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.head_dim, 1),
        #     nn.ReLU())
        # self.sigma = nn.Sequential(nn.Linear(self.head_dim, 1), nn.Softplus())
        self.color_gs = nn.Sequential(
            nn.Linear(self.head_dim+3, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 3),
            nn.Sigmoid()
        )
        
    def forward(self, sigma, radiance, nerf_feats, masks, z_vals, batch, size, level):
        
        # from here
        B, K, N_rays, N_samples = sigma.shape
        # print(sigma.shape)
        H, W = size
        masks = masks.view(B, K, N_rays, N_samples)

        raw2alpha = lambda raw: 1.-torch.exp(-raw)
        alpha_all = raw2alpha(sigma)  # [B, K, N_rays, N_samples] 
        alphas = torch.sum(alpha_all * masks, dim=1) # [B, N_rays, N_samples]

        T_full = torch.cumprod(1.-alphas+1e-10, dim=-1)[..., :-1]
        T_full = torch.cat([torch.ones_like(alphas[..., 0:1]), T_full], dim=-1) # # [B, N_rays, N_samples]
        # T_full = torch.cumprod(torch.cat([torch.ones_like(alphas[..., 0:1]), 1-alphas], dim=-1), dim=-1)[..., :-1] # [B, N_rays, N_samples]
    
        weights_full = alphas * T_full # [B, N_rays, N_samples]
        
        radiance = radiance.reshape(B,K,N_rays,N_samples,3) # [B, K, N_rays, N_samples, 3]
        # print(rgb_all.shape)
        rgb_vr_full = torch.sum((T_full.unsqueeze(1).repeat(1, K, 1, 1)*alpha_all*masks)[...,None] * radiance, -2)  # [B, K, N_rays, 3]
        rgb_vr_full = torch.sum(rgb_vr_full, dim=1) # [B, N_rays, 3]
        rgb_vr_full = rgb_vr_full.reshape(B,H,W,3).permute(0,3,1,2) # [B, 3, H, W]

        # depth  check here
        d = z_vals.shape[-1]
        if z_vals is not None:
            weights_full = F.softmax(weights_full, dim=-1)
            depth_vr_full = torch.sum(weights_full*z_vals.detach().mean(1), -1)
            depth_vr_full = depth_vr_full.reshape(B,H,W)# [B, H, W]
        else:
            depth_vr_full = None

        # world_xyz
        # weights = weights.reshape(B,H,W,d).permute(0,3,1,2)
        # depth = torch.sum(weights * depth, 1) # B H W
        # # depth = torch.sum(depth, 1) # B H W
        ext = batch['tar_ext']
        ixt = batch['tar_ixt'].clone()
        ixt[:,:2] *= cfg.credsplatting.cas_config.render_scale[level]
        world_xyz = depth2point(depth_vr_full, ext, ixt)


        # todo: feature fuse
        # nerf_feats # [B, K, N_rays, 24] 考虑加个attention 融合K个特征
        # masks # [B, K, N_rays, d]
        fuse_nerf_feats = torch.sum(nerf_feats*masks, 1)  # [B, K, N_rays, 24] -> [B, N_rays, 24]

        x = self.lr0(fuse_nerf_feats) # [B, N_rays, 24]
        # enhance features using a UNet
        x = x.reshape(B,H*W,d,x.shape[-1])  # [B, N_rays, d, 24]
        # print(x.shape, weights.shape)
        x = torch.sum(weights_full[...,None].unsqueeze(0) * x, -2)
        x = x.reshape(B,H,W,x.shape[-1]).permute(0,3,1,2)
        x = self.Unet(x)
        x = x.flatten(-2).permute(0,2,1)

        # gs branch
        # rot head
        # rot_out = torch.ones((B,x.shape[1], 4)).to(x.device)
        rot_out = self.rotation_head(x)
        rot_out = torch.nn.functional.normalize(rot_out, dim=-1)
      
        # scale head
        scale_out = self.scale_head(x)
        
        # opacity head
        opacity_out = self.opacity_head(x)
        
        # color head
        x0 = torch.cat((x, rgb_vr_full.flatten(2).permute(0,2,1)),dim=-1)
        color_out = self.color_gs(x0)
        
        
        return world_xyz, rot_out, scale_out, opacity_out, color_out, rgb_vr_full, depth_vr_full





class GS4_fuse(nn.Module):
    def __init__(self, hid_n=64, feat_ch=16+3):
        """
        """
        super(GS4_fuse, self).__init__()
        self.hid_n = hid_n
        self.agg = Agg(feat_ch)
        self.head_dim = 24

        self.lr0 = nn.Sequential(nn.Linear(8+16, self.head_dim),
                                 nn.ReLU())
        self.lrs = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_n, hid_n), nn.ReLU()) for i in range(0)])
        self.sigma = nn.Sequential(nn.Linear(self.head_dim, 1), nn.Softplus())
        self.color = nn.Sequential(
            nn.Linear(feat_ch+self.head_dim+4, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 1),
            nn.ReLU())
        
        self.Unet = Unet(self.head_dim, 16)
        self.opacity_head = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 1),
            nn.Sigmoid()
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 4),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 3),
            nn.Softplus()
        )

        self.color_gs = nn.Sequential(
            nn.Linear(self.head_dim+3, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 3),
            nn.Sigmoid()
        )
        
    def forward(self, vox_feat, img_feat_rgb_dir, z_vals, batch, size, level):
        H,W = size
        B, N_points, N_views = img_feat_rgb_dir.shape[:-1]
        S = img_feat_rgb_dir.shape[2] # 3
        img_feat = self.agg(img_feat_rgb_dir)
        vox_img_feat = torch.cat((vox_feat, img_feat), dim=-1)  # vox_img_feat [1, 327680, 24]
        
        feature_nerf = vox_img_feat.clone()
        
        # depth 
        d = z_vals.shape[-1]
        z_vals = z_vals.reshape(B,H,W,d)
        if cfg.credsplatting.cas_config.depth_inv[level]:
            z_vals = 1./torch.clamp_min(z_vals, 1e-6) # to disp
        depth = z_vals.permute(0,3,1,2)

        # radiance head
        x = self.lr0(vox_img_feat) # [B, N_rays, 24]
        sigma = self.sigma(x) # [B, N_rays, 1]
        # x = torch.cat((x, vox_img_feat), dim=-1) # [B, N_rays, 48]

        # x = x.view(B, -1, 1, x.shape[-1]).repeat(1, 1, S, 1)
        x0 = x.unsqueeze(2).repeat(1,1,S,1)   # [1, N_rays, 3, 24]
        x0 = torch.cat((x0, img_feat_rgb_dir), dim=-1) # [1, N_rays, 3, 63]
        color_weight = F.softmax(self.color(x0), dim=-2) # [1, N_rays, 3, 1]
        # print(img_feat_rgb_dir[..., -7:-4].shape)        # [1, N_rays, 3, 3]
        radiance = torch.sum((img_feat_rgb_dir[..., -7:-4] * color_weight), dim=-2) # [1, N_rays, 3]

        # volume rendering branch
        sigma0 = sigma.reshape(B,H*W,d)
        raw2alpha = lambda raw: 1.-torch.exp(-raw)
        alpha = raw2alpha(sigma0)  # [N_rays, N_samples]
        T = torch.cumprod(1.-alpha+1e-10, dim=-1)[..., :-1]
        T = torch.cat([torch.ones_like(alpha[..., 0:1]), T], dim=-1)
        weights = alpha * T
        radiance0 = radiance.reshape(B,H*W,d,3)
        rgb_vr = torch.sum(weights[...,None] * radiance0, -2) 
        rgb_vr = rgb_vr.reshape(B,H,W,3).permute(0,3,1,2)


        # enhance features using a UNet
        x = x.reshape(B,H*W,d,x.shape[-1])
        x = torch.sum(weights[...,None].unsqueeze(0) * x, -2) 
        x = x.reshape(B,H,W,x.shape[-1]).permute(0,3,1,2)
        x = self.Unet(x)
        x = x.flatten(-2).permute(0,2,1)
        # print(x.shape)

        feature_gs = x.clone()
        # fea_transmit = torch.cat((x, vox_feat, img_feat), dim=-1)
        
        # gs branch
        # rot head
        rot_out = torch.ones((B,x.shape[1],4)).to(x.device)
        # rot_out = self.rotation_head(x)
        rot_out = torch.nn.functional.normalize(rot_out, dim=-1)
      
        # scale head
        scale_out = self.scale_head(x)  # 3
        
        # opacity head
        opacity_out = self.opacity_head(x)
        
        # color head
        x0 = torch.cat((x,rgb_vr.flatten(2).permute(0,2,1)),dim=-1)
        # x0 = torch.cat((x,radiance),dim=-1)
        color_out = self.color_gs(x0)
        

        return radiance, sigma, rot_out, scale_out, opacity_out, color_out, feature_nerf, feature_gs




class GS_agg1(nn.Module):
    def __init__(self):
        """
        """
        super(GS_agg1, self).__init__()
        self.head_dim = 16

        # self.Unet = Unet(self.head_dim, 16)
        self.opacity_head = nn.Sequential(
            nn.Linear(1, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 1),
            nn.Sigmoid()
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(4, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 4),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(3, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 3),
            nn.Softplus()
        )

        self.color_gs = nn.Sequential(
            nn.Linear(3, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 3),
            nn.Sigmoid()
        )
        
    def forward(self, rot, scale, opacity, color):

        rot_out = self.rotation_head(rot)
        rot_out = torch.nn.functional.normalize(rot_out, dim=-1)
        # print(rot_out.shape)
      
        # scale head
        scale_out = self.scale_head(scale)  # 3
        # print(scale_out.shape)
        
        # opacity head
        opacity_out = self.opacity_head(opacity)
        # print(opacity_out.shape)
        
        # color head
        color_out = self.color_gs(color)
        # print(color_out.shape)
        
        return rot_out, scale_out, opacity_out, color_out



class Agg(nn.Module):
    def __init__(self, feat_ch):
        super(Agg, self).__init__()
        self.feat_ch = feat_ch
        if cfg.credsplatting.viewdir_agg:
            self.view_fc = nn.Sequential(
                    nn.Linear(4, feat_ch),
                    nn.ReLU(),
                    )
            self.view_fc.apply(weights_init)
        self.global_fc = nn.Sequential(
                nn.Linear(feat_ch*3, 32),
                nn.ReLU(),
                )

        self.agg_w_fc = nn.Sequential(
                nn.Linear(32, 1),
                nn.ReLU(),
                )
        self.fc = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                )
        self.global_fc.apply(weights_init)
        self.agg_w_fc.apply(weights_init)
        self.fc.apply(weights_init)

    def forward(self, img_feat_rgb_dir):
        B, S = len(img_feat_rgb_dir), img_feat_rgb_dir.shape[-2]
        if cfg.credsplatting.viewdir_agg:
            view_feat = self.view_fc(img_feat_rgb_dir[..., -4:])
            img_feat_rgb =  img_feat_rgb_dir[..., :-4] + view_feat
        else:
            img_feat_rgb =  img_feat_rgb_dir[..., :-4]

        var_feat = torch.var(img_feat_rgb, dim=-2).view(B, -1, 1, self.feat_ch).repeat(1, 1, S, 1)
        avg_feat = torch.mean(img_feat_rgb, dim=-2).view(B, -1, 1, self.feat_ch).repeat(1, 1, S, 1)

        feat = torch.cat([img_feat_rgb, var_feat, avg_feat], dim=-1)
        global_feat = self.global_fc(feat)
        agg_w = F.softmax(self.agg_w_fc(global_feat), dim=-2)
        im_feat = (global_feat * agg_w).sum(dim=-2)
        return self.fc(im_feat)



def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

