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
from PIL import Image
import torchvision.transforms as T


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

            # print(batch[f'msk_{i}'].shape, h , w)
            masks = (batch[f'msk_{i}'].reshape(B, h, w).cpu().numpy() >= 1).astype(np.uint8)

            if cfg.credsplatting.eval_center:
                H_crop, W_crop = int(h*0.1), int(w*0.1)
                pred_rgb = pred_rgb[:, H_crop:-H_crop, W_crop:-W_crop]
                gt_rgb = gt_rgb[:, H_crop:-H_crop, W_crop:-W_crop]
                masks = masks[:, H_crop:-H_crop, W_crop:-W_crop]

            for b in range(B):
                if not batch['meta']['scene'][b]+f'_level{i}' in self.scene_psnrs:
                    self.scene_psnrs[batch['meta']['scene'][b]+f'_level{i}'] = []
                    self.scene_ssims[batch['meta']['scene'][b]+f'_level{i}'] = []
                    self.scene_lpips[batch['meta']['scene'][b]+f'_level{i}'] = []
                if cfg.save_result and i == cfg.credsplatting.cas_config.num-1:
                    # img = img_utils.horizon_concate(gt_rgb[b], pred_rgb[b])
                    # img_path = os.path.join(cfg.result_dir, '{}_{}_{}.png'.format(batch['meta']['scene'][b], batch['meta']['tar_view'][b].item(), batch['meta']['frame_id'][b].item()))
                    # # print(img_path)
                    # # print(gt_rgb[b].shape)
                    # imageio.imwrite(img_path, (img * 255.).astype(np.uint8))

                    gt_img = gt_rgb[b]
                    pred_img = pred_rgb[b]
                    # 保存 gt_rgb 图像
                    gt_img_path = os.path.join(cfg.result_dir, '{}_{}_{}_gt.png'.format(batch['meta']['scene'][b], batch['meta']['tar_view'][b].item(), batch['meta']['frame_id'][b].item()))
                    # print(gt_img_path)
                    imageio.imwrite(gt_img_path, (gt_img * 255.).astype(np.uint8))

                    # 保存 pred_rgb 图像
                    pred_img_path = os.path.join(cfg.result_dir, '{}_{}_{}_{}_pred.png'.format(batch['meta']['scene'][b], batch['meta']['tar_view'][b].item(), batch['meta']['frame_id'][b].item(), cfg.credsplatting.test_input_views))
                    imageio.imwrite(pred_img_path, (pred_img * 255.).astype(np.uint8))



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

                # print(1234)
                # print(cfg.credsplatting.eval_depth, i)
                # if cfg.credsplatting.eval_depth and (i == cfg.credsplatting.cas_config.num - 1) and batch['meta']['scene'][b] in self.eval_depth_scenes:
                if cfg.credsplatting.eval_depth and (i == cfg.credsplatting.cas_config.num - 1):
                    mvs_depth = output[f'depth_mvs2_level{i}'].cpu().numpy()[b]
                    # mvs_depth = output[f'depth_mvs2_level{i}'].cpu().numpy()[b]
                    # nerf_gt_depth = batch['tar_dpt'][b].cpu().numpy().reshape((h, w))
                    # mvs_gt_depth = cv2.resize(nerf_gt_depth, mvs_depth.shape[::-1], interpolation=cv2.INTER_NEAREST)
                    # mvs_mask = np.logical_and(mvs_gt_depth > 425., mvs_gt_depth < 905.)
                    # mvs_mask = mvs_gt_depth != 0.
                    # self.mvs_abs.append((np.abs(mvs_depth[mvs_mask] - mvs_gt_depth[mvs_mask])).mean())
                    # self.mvs_acc_2.append((np.abs(mvs_depth[mvs_mask] - mvs_gt_depth[mvs_mask]) < 2.).mean())
                    # self.mvs_acc_10.append((np.abs(mvs_depth[mvs_mask] - mvs_gt_depth[mvs_mask]) < 10.).mean())
                    # print(1111111, mvs_depth.shape)
                    # if cfg.credsplatting.eval_center:
                    #     h, w = mvs_depth.shape
                    #     H_crop, W_crop = int(h*0.1), int(w*0.1)
                    #     mvs_depth = mvs_depth[H_crop:-H_crop, W_crop:-W_crop]
                        # gt_rgb = gt_rgb[H_crop:-H_crop, W_crop:-W_crop]
                        # masks = masks[H_crop:-H_crop, W_crop:-W_crop]

                    # 计算 gt_depth 和 pred_depth
                    # gt_depth_img = mvs_gt_depth * mvs_mask
                    # pred_depth_img = mvs_depth * mvs_mask
                    # gt_depth_img = mvs_gt_depth
                    pred_depth_img = mvs_depth

                    # 保存 gt_depth 图像
                    # gt_depth_img_path = os.path.join(cfg.result_dir, 'depth_{}_{}_{}_gt.png'.format(batch['meta']['scene'][b], batch['meta']['tar_view'][b].item(), batch['meta']['frame_id'][b].item()))
                    # colormap = cm.get_cmap('jet').reversed()
                    # # # 将 gt_depth 映射到彩色图 (0-255)
                    # # gt_depth_img = np.clip(gt_depth_img, 425., 905.)
                    # # gt_depth_color = colormap((gt_depth_img - 425.) / (905. - 425.))[:, :, :3]  # 去掉 alpha 通道
                    # # imageio.imwrite(gt_depth_img_path, (gt_depth_color * 255.).astype(np.uint8))

                    # 保存 pred_depth 图像
                    pred_depth_img_path = os.path.join(cfg.result_dir, '{}_{}_{}_pred_depth.png'.format(batch['meta']['scene'][b], batch['meta']['tar_view'][b].item(), batch['meta']['frame_id'][b].item()))
                    # 将 pred_depth 映射到彩色图 (0-255)
                    # pred_depth_img = np.clip(pred_depth_img, 425., 905.)
                    # pred_depth_color = colormap((pred_depth_img - pred_depth_img.min()) / (pred_depth_img.max() - pred_depth_img.min()))[:, :, :3]  # 去掉 alpha 通道
                    mi, ma = batch['near_far'].min().detach().cpu().numpy(), batch['near_far'].max().detach().cpu().numpy()
                    pred_depth_img = (pred_depth_img - mi) / (ma - mi + 1e-8)  # normalize to 0~1
                    pred_depth_img = (255 * pred_depth_img).astype(np.uint8)
                    pred_depth_img = Image.fromarray(cv2.applyColorMap(pred_depth_img, cv2.COLORMAP_JET))
                    pred_depth_img = T.ToTensor()(pred_depth_img)  # (3, H, W)
                    pred_depth_color = pred_depth_img.permute(1,2,0).detach().cpu().numpy()

                    
                    if cfg.credsplatting.eval_center:
                        h, w = mvs_depth.shape
                        H_crop, W_crop = int(h*0.1), int(w*0.1)
                        pred_depth_color = pred_depth_color[H_crop:-H_crop, W_crop:-W_crop,:]
                    # print(pred_depth_color.shape)
                    imageio.imwrite(pred_depth_img_path, (pred_depth_color * 255.).astype(np.uint8))

                    # 定义伽马值
                    # gamma = 0.5  # 你可以调整这个值来控制区分度，0 < gamma < 1 会使低值更亮，大于1 会使高值更亮

                    # # 保存 pred_depth 图像
                    # pred_depth_img_path = os.path.join(cfg.result_dir, 'depth_{}_{}_{}_pred.png'.format(
                    #     batch['meta']['scene'][b], batch['meta']['tar_view'][b].item(), batch['meta']['frame_id'][b].item()))

                    # # 将 pred_depth 映射到彩色图 (0-255)
                    # pred_depth_img = np.clip(pred_depth_img, 425., 905.)

                    # # 归一化深度值到 [0, 1] 范围
                    # norm_pred_depth = (pred_depth_img - pred_depth_img.min()) / (pred_depth_img.max() - pred_depth_img.min())

                    # # 应用伽马变换以增加区分度
                    # gamma_corrected = np.power(norm_pred_depth, gamma)

                    # # 使用 colormap 将处理后的深度值映射为 RGB 颜色
                    # colormap = cm.get_cmap('viridis')  # 你可以选择其他 colormap
                    # pred_depth_color = colormap(gamma_corrected)[:, :, :3]  # 只取 RGB 通道，不要 alpha 通道

                    # # 将颜色映射值转换为 [0, 255] 之间的整数
                    # pred_depth_color_8bit = (pred_depth_color * 255.).astype(np.uint8)

                    # # 保存为 PNG 图像
                    # imageio.imwrite(pred_depth_img_path, pred_depth_color_8bit)


                    

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
