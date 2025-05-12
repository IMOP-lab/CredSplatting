import numpy as np
from lib.config import cfg
import os
import imageio
from lib.utils import img_utils
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import lpips
import imageio
import cv2
import logging
import matplotlib.cm as cm


class Evaluator:

    def __init__(self,):
        self.psnrs = []
        self.ssims = []
        self.lpips = []
        self.scene_psnrs = {}
        self.scene_ssims = {}
        self.scene_lpips = {}
        self.loss_fn_vgg = lpips.LPIPS(net='vgg')
        self.loss_fn_vgg.cuda()
        if cfg.credsplatting.eval_depth:
            # Following the setup of MVSNeRF
            self.eval_depth_scenes = ['scan1', 'scan8', 'scan21', 'scan103', 'scan110']
            self.abs = []
            self.acc_2 = []
            self.acc_10 = []
            self.mvs_abs = []
            self.mvs_acc_2 = []
            self.mvs_acc_10 = []
        os.system('mkdir -p ' + cfg.result_dir)

    def evaluate(self, output, batch):
        B, S, _, H, W = batch['src_inps'].shape
        for i in range(cfg.credsplatting.cas_config.num):
            if not cfg.credsplatting.cas_config.render_if[i]:
                continue
            render_scale = cfg.credsplatting.cas_config.render_scale[i]
            h, w = int(H*render_scale), int(W*render_scale)
            pred_rgb = output[f'rgb_level{i}'].reshape(B, h, w, 3).detach().cpu().numpy()
            gt_rgb   = batch[f'rgb_{i}'].reshape(B, h, w, 3).detach().cpu().numpy()

            masks = (batch[f'msk_{i}'].reshape(B, h, w).cpu().numpy() >= 1).astype(np.uint8)
            
            src_imgs = batch[f'src_inps'].permute(0,1,3,4,2).detach().cpu().numpy()
            
            
            if cfg.credsplatting.eval_center:
                H_crop, W_crop = int(h*0.1), int(w*0.1)
                pred_rgb = pred_rgb[:, H_crop:-H_crop, W_crop:-W_crop]
                gt_rgb = gt_rgb[:, H_crop:-H_crop, W_crop:-W_crop]
                masks = masks[:, H_crop:-H_crop, W_crop:-W_crop]
                src_imgs = src_imgs[:, H_crop:-H_crop, W_crop:-W_crop]
            
            
            for b in range(B):
                if not batch['meta']['scene'][b]+f'_level{i}' in self.scene_psnrs:
                    self.scene_psnrs[batch['meta']['scene'][b]+f'_level{i}'] = []
                    self.scene_ssims[batch['meta']['scene'][b]+f'_level{i}'] = []
                    self.scene_lpips[batch['meta']['scene'][b]+f'_level{i}'] = []
                if cfg.save_result and i == cfg.credsplatting.cas_config.num-1:
                    # img = img_utils.horizon_concate(gt_rgb[b], pred_rgb[b])
                    # img_path = os.path.join(cfg.result_dir, '{}_{}_{}_{}.png'.format(batch['meta']['scene'][b], batch['meta']['tar_view'][b].item(), batch['meta']['frame_id'][b].item(), i))
                    # imageio.imwrite(img_path, (img * 255.).astype(np.uint8))
                    src_list = []
                    # print(src_imgs[b].shape[0])
                    for v in range(src_imgs[b].shape[0]):
                        src_list.append(src_imgs[b][v,:,:,:] * 0.5 + 0.5)
                    src_list.append(gt_rgb[b])
                    src_list.append(pred_rgb[b])
                    img = grid_concate2(src_list, 2, src_imgs[b].shape[0])
                    img_path = os.path.join(cfg.result_dir, '{}_{}_{}_{}.png'.format(batch['meta']['scene'][b], batch['meta']['tar_view'][b].item(), batch['meta']['frame_id'][b].item(), i))
                    imageio.imwrite(img_path, (img * 255.).astype(np.uint8))

                mask = masks[b] == 1
                gt_rgb[b][mask==False] = 0.
                pred_rgb[b][mask==False] = 0.

                psnr_item = psnr(gt_rgb[b][mask], pred_rgb[b][mask], data_range=1.)
                if i == cfg.credsplatting.cas_config.num-1:
                    self.psnrs.append(psnr_item)
                self.scene_psnrs[batch['meta']['scene'][b]+f'_level{i}'].append(psnr_item)

                ssim_item = ssim(gt_rgb[b], pred_rgb[b], multichannel=True)
                if i == cfg.credsplatting.cas_config.num-1:
                    self.ssims.append(ssim_item)
                self.scene_ssims[batch['meta']['scene'][b]+f'_level{i}'].append(ssim_item)

                if cfg.eval_lpips:
                    gt, pred = torch.Tensor(gt_rgb[b])[None].permute(0, 3, 1, 2), torch.Tensor(pred_rgb[b])[None].permute(0, 3, 1, 2)
                    gt, pred = (gt-0.5)*2., (pred-0.5)*2.
                    lpips_item = self.loss_fn_vgg(gt.cuda(), pred.cuda()).item()
                    if i == cfg.credsplatting.cas_config.num-1:
                        self.lpips.append(lpips_item)
                    self.scene_lpips[batch['meta']['scene'][b]+f'_level{i}'].append(lpips_item)

                # if cfg.credsplatting.eval_depth and (i == cfg.credsplatting.cas_config.num - 1) and batch['meta']['scene'][b] in self.eval_depth_scenes:
                if cfg.credsplatting.eval_depth and (i == cfg.credsplatting.cas_config.num - 1):
                    mvs_depth = output[f'depth_mvs_level{i}'].cpu().numpy()[b]
                    nerf_gt_depth = batch['tar_dpt'][b].cpu().numpy().reshape((h, w))
                    
                    nerf_gs_depth = output[f'depth_gs_level{i}'].permute(0,2,3,1).cpu().numpy()[b]
                    nerf_gs_depth = cv2.resize(nerf_gs_depth, mvs_depth.shape[::-1], interpolation=cv2.INTER_NEAREST)
                    # print(output[f'depth_mvs2_level{i}'].shape)
                    mvs_depth2 = output[f'depth_mvs2_level{i}'].cpu().numpy()[b]
                    # print(mvs_depth2.shape)
                    mvs_depth2 = cv2.resize(mvs_depth2, mvs_depth.shape[::-1], interpolation=cv2.INTER_NEAREST)

                    mvs_gt_depth = cv2.resize(nerf_gt_depth, mvs_depth.shape[::-1], interpolation=cv2.INTER_NEAREST)
                    mvs_mask = np.logical_and(mvs_gt_depth > 425., mvs_gt_depth < 905.)
                    mvs_mask = mvs_gt_depth != 0.
                    self.mvs_abs.append((np.abs(mvs_depth[mvs_mask] - mvs_gt_depth[mvs_mask])).mean())
                    self.mvs_acc_2.append((np.abs(mvs_depth[mvs_mask] - mvs_gt_depth[mvs_mask]) < 2.).mean())
                    self.mvs_acc_10.append((np.abs(mvs_depth[mvs_mask] - mvs_gt_depth[mvs_mask]) < 10.).mean())
                    
                    # print(mvs_depth.shape)
                    dep_img = [mvs_gt_depth * mvs_mask, mvs_depth * mvs_mask, mvs_depth2 * mvs_mask, nerf_gs_depth * mvs_mask]
                    # img = img_utils.horizon_concate(np.expand_dims(mvs_depth * mvs_mask, axis=-1), np.expand_dims(mvs_gt_depth * mvs_mask, axis=-1))
                    # img = img_utils.horizon_concate(mvs_gt_depth * mvs_mask, mvs_depth * mvs_mask)
                    img = grid_concate2(dep_img, 2, 2)
                    img_path = os.path.join(cfg.result_dir, 'depth_{}_{}_{}_{}.png'.format(batch['meta']['scene'][b], batch['meta']['tar_view'][b].item(), batch['meta']['frame_id'][b].item(), i))
                    colormap = cm.get_cmap('hot')
                    # 将深度图映射到彩色图 (0-255)
                    # print(np.unique(nerf_gs_depth))
                    depth_color = colormap((img - 425.) / (905.-425.))[:, :, :3]  # 去掉 alpha 通道
                    # depth_color = colormap(img / 1000. )[:, :, :3]  # 去掉 alpha 通道
                    # print(depth_color.shape)
                    # print(img_path)
                    imageio.imwrite(img_path, (depth_color * 255.).astype(np.uint8))
                    
                    
                    

                    

    def summarize(self):
        ret = {}
        ret.update({'psnr': np.mean(self.psnrs)})
        ret.update({'ssim': np.mean(self.ssims)})
        if cfg.eval_lpips:
            ret.update({'lpips': np.mean(self.lpips)})
        print('='*30)
        logging.info('='*30)
        for scene in self.scene_psnrs:
            if cfg.eval_lpips:
                # print(1234)
                print(scene.ljust(16), 'psnr: {:.2f} ssim: {:.3f} lpips:{:.3f}'.format(np.mean(self.scene_psnrs[scene]), np.mean(self.scene_ssims[scene]), np.mean(self.scene_lpips[scene])))
                # logging.info('{} psnr: {:.2f} ssim: {:.3f} lpips:{:.3f}'.format(scene.ljust(16), np.mean(self.scene_psnrs[scene]), np.mean(self.scene_ssims[scene]), np.mean(self.scene_lpips[scene])))
            else:
                # print(1234)
                print(scene.ljust(16), 'psnr: {:.2f} ssim: {:.3f} '.format(np.mean(self.scene_psnrs[scene]), np.mean(self.scene_ssims[scene])))
                # logging.info('{} psnr: {:.2f} ssim: {:.3f} '.format(scene.ljust(16), np.mean(self.scene_psnrs[scene]), np.mean(self.scene_ssims[scene])))
        print('='*30)
        logging.info('='*30)
        print(ret)
        logging.info(ret)
        if cfg.credsplatting.eval_depth:            
            keys = ['mvs_abs', 'mvs_acc_2', 'mvs_acc_10']
            depth_ret = {}
            for key in keys:
                depth_ret[key] = np.mean(getattr(self, key))
                setattr(self, key, [])
            print(depth_ret)
            logging.info(depth_ret)
        self.psnrs = []
        self.ssims = []
        self.lpips = []
        self.scene_psnrs = {}
        self.scene_ssims = {}
        self.scene_lpips = {}
        if cfg.save_result:
            print('Save visualization results to: {}'.format(cfg.result_dir))
            logging.info('Save visualization results to: {}'.format(cfg.result_dir))
        return ret



def grid_concate2(images, rows, cols):
    # 确保输入的图片数量符合要求
    if not isinstance(images, list) or len(images) < 1:
        raise ValueError("Input should be a list of images.")
    
    # 计算网格中每个格子的最大宽度和高度
    max_height = max(img.shape[0] for img in images)  # 找到所有图片中最高的高度
    max_width = max(img.shape[1] for img in images)  # 找到所有图片中最宽的宽度

    # 计算最终拼接图像的总高度和总宽度
    total_height = rows * max_height
    total_width = cols * max_width

    # 如果图片是彩色图片（即包含RGB通道），初始化黑色背景的拼接图像
    if images[0].ndim == 3:  # 彩色图像
        concatenated_image = np.zeros((total_height, total_width, 3), dtype=images[0].dtype)
    else:  # 灰度图像
        concatenated_image = np.zeros((total_height, total_width), dtype=images[0].dtype)

    # 将图片逐一放入网格中
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break  # 如果图片超过给定的网格数，就停止
        h, w = img.shape[:2]
        row_idx = idx // cols  # 计算当前图片的行索引
        col_idx = idx % cols   # 计算当前图片的列索引

        # 计算在最终图片中的起始位置
        start_y = row_idx * max_height
        start_x = col_idx * max_width

        # 将图片放入合适的位置
        if img.ndim == 3:
            concatenated_image[start_y:start_y + h, start_x:start_x + w, :] = img
        else:
            concatenated_image[start_y:start_y + h, start_x:start_x + w] = img

    return concatenated_image
