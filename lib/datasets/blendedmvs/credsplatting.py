import numpy as np
import os
from lib.datasets import credsplatting_utils
from lib.config import cfg
import imageio
import cv2
import random
from lib.utils import data_utils
import torch
from lib.utils.video_utils import *
import sys

if cfg.fix_random:
    random.seed(0)
    np.random.seed(0)

class Dataset:
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = os.path.join(cfg.workspace, kwargs['data_root'])
        self.split = kwargs['split']
        self.input_h_w = kwargs['input_h_w']
        if 'scene' in kwargs:
            self.scenes = [kwargs['scene']]
        else:
            self.scenes = []
        self.scale_factor = 100.
        self.build_metas()
        # self.depth_ranges = [425., 905.]
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = [0.0, 0.0, 0.0]
        self.scale = 1.0

    def build_metas(self):
        # scenes = [line.strip() for line in open(ann_file).readlines()]
        # dtu_pairs = torch.load('data/credsplatting/pairs.th') 
        scenes = [name for name in os.listdir(self.data_root) if not os.path.isfile(os.path.join(self.data_root, name))]
        # print(scenes)
        self.scene_infos = {}
        self.metas = []
        if len(self.scenes) != 0:
            scenes = self.scenes

        for scene in scenes:
            scene_info = {'ixts': [], 'exts': [], 'dpt_paths': [], 'img_paths': [], 'depth_range': [], "depth_range_each": []}
            cam_root = os.path.join(self.data_root, scene, 'cams')
            cam_list = [name for name in os.listdir(cam_root) if name[-8:] == '_cam.txt']
            all_depth_min, all_depth_max = 100000, -1
            for i in range(len(cam_list)):
                base_name = cam_list[i].split('_cam')[0]
                cam_path = os.path.join(cam_root, cam_list[i])
                ixt, ext, line11 = data_utils.read_cam_file(cam_path)
                # depth range
                depth_min = line11[0]
                # depth_interval = line11[1]
                # depth_num = line11[2]
                depth_max = line11[3]
                # depth_num = int((depth_max - depth_min) / depth_interval / 32.0 + 1) * 32
                # depth_max = depth_min + float(depth_num) * depth_interval
                # print(depth_min, depth_max)
                all_depth_max = max(depth_max, all_depth_max)
                all_depth_min = min(depth_min, all_depth_min)
                ext[:3, 3] = ext[:3, 3] * self.scale_factor
                ixt[:2] = ixt[:2] * 1
                dpt_path = os.path.join(self.data_root, scene, f'rendered_depth_maps/{base_name}.pfm')
                img_path = os.path.join(self.data_root, scene, f'blended_images/{base_name}_masked.jpg')

                scene_info['ixts'].append(ixt.astype(np.float32))
                scene_info['exts'].append(ext.astype(np.float32))
                scene_info['dpt_paths'].append(dpt_path)
                scene_info['img_paths'].append(img_path)
                scene_info['depth_range_each'].append([depth_min, depth_max])
            # scene_info['depth_range']=[all_depth_min, all_depth_max]

            img_len = len(scene_info['img_paths'])
            # render_ids = [8]
            render_ids = [j for j in range(img_len // 8, img_len, 8)]
            train_ids = [j for j in range(img_len) if j not in render_ids]
            scene_info.update({'train_ids': train_ids, 'test_ids': render_ids})
            self.scene_infos[scene] = scene_info
            # print(len(render_ids), len(train_ids))

            cam_points = np.array([np.linalg.inv(scene_info['exts'][i])[:3, 3] for i in train_ids])
            for tar_view in render_ids:
                cam_point = np.linalg.inv(scene_info['exts'][tar_view])[:3, 3]
                distance = np.linalg.norm(cam_points - cam_point[None], axis=-1)
                argsorts = distance.argsort()
                argsorts = argsorts[1:] if tar_view in train_ids else argsorts
                input_views_num = cfg.credsplatting.train_input_views[2] + 1 if self.split == 'train' else cfg.credsplatting.test_input_views
                src_views = [train_ids[i] for i in argsorts[:input_views_num]]
                self.metas += [(scene, tar_view, src_views)]
            break
    
    def __getitem__(self, index_meta):
        index, input_views_num = index_meta
        scene, tar_view, src_views = self.metas[index]
        # print(index_meta, 'BlendedMVS')
        
        if self.split == 'train':
            if random.random() < 0.1:
                src_views = src_views + [tar_view]
            src_views = random.sample(src_views[:input_views_num+1], input_views_num)
        scene_info = self.scene_infos[scene]
        # print(src_views)

        tar_img = np.array(imageio.imread(scene_info['img_paths'][tar_view])) / 255.
        tar_ext, tar_ixt = scene_info['exts'][tar_view], scene_info['ixts'][tar_view]
        orig_size = tar_img.shape[:2][::-1]
        tar_img = cv2.resize(tar_img, self.input_h_w[::-1], interpolation=cv2.INTER_AREA)
        tar_ixt[0] *= self.input_h_w[1] / orig_size[0]
        tar_ixt[1] *= self.input_h_w[0] / orig_size[1]

        H, W = tar_img.shape[:2]

        if self.split != 'train': # only used for evaluation
            tar_dpt = data_utils.read_pfm(scene_info['dpt_paths'][tar_view])[0].astype(np.float32)
            tar_dpt = cv2.resize(tar_dpt, [H, W], fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            tar_mask = (tar_dpt > 0.).astype(np.uint8)
        else:
            tar_dpt = np.ones_like(tar_img)
            tar_mask = np.ones_like(tar_img)

        src_inps, src_exts, src_ixts = self.read_src(scene_info, src_views)
        # print(src_exts, src_ixts)

        ret = {'src_inps': src_inps,
               'src_exts': src_exts,
               'src_ixts': src_ixts}
        ret.update({'tar_ext': tar_ext,
                    'tar_ixt': tar_ixt})
        # if self.split != 'train':
        ret.update({'tar_img': tar_img,
                    'tar_dpt': tar_dpt,
                    'tar_mask': tar_mask})
        depth_range = scene_info['depth_range_each'][tar_view]
        # print(depth_range)
        ret.update({'near_far': np.array([depth_range[0]* self.scale_factor, depth_range[1] * self.scale_factor]).astype(np.float32)})
        ret.update({'meta': {'scene': scene, 'tar_view': tar_view, 'frame_id': 0}})

        for i in range(cfg.credsplatting.cas_config.num):
            rays, rgb, msk = credsplatting_utils.build_rays(tar_img, tar_ext, tar_ixt, tar_mask, i, self.split)
            s = cfg.credsplatting.cas_config.volume_scale[i]
            # if self.split != 'train': # evaluation
            #     tar_dpt_i = cv2.resize(tar_dpt, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            #     ret.update({f'tar_dpt_{i}': tar_dpt_i.astype(np.float32)})
            ret.update({f'rays_{i}': rays, f'rgb_{i}': rgb.astype(np.float32), f'msk_{i}': msk})
            ret['meta'].update({f'h_{i}': int(H*s), f'w_{i}': int(W*s)})
            
        R = np.array(tar_ext[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(tar_ext[:3, 3], np.float32)
        for i in range(cfg.credsplatting.cas_config.num):
            h, w = H*cfg.credsplatting.cas_config.render_scale[i], W*cfg.credsplatting.cas_config.render_scale[i]
            tar_ixt_ = tar_ixt.copy()
            tar_ixt_[:2,:] *= cfg.credsplatting.cas_config.render_scale[i]
            FovX = data_utils.focal2fov(tar_ixt_[0, 0], w)
            FovY = data_utils.focal2fov(tar_ixt_[1, 1], h)
            projection_matrix = data_utils.getProjectionMatrix(znear=self.znear, zfar=self.zfar, K=tar_ixt_, h=h, w=w).transpose(0, 1)
            world_view_transform = torch.tensor(data_utils.getWorld2View2(R, T, np.array(self.trans), self.scale)).transpose(0, 1)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]
            novel_view_data = {
                'FovX':  torch.FloatTensor([FovX]),
                'FovY':  torch.FloatTensor([FovY]),
                'width': w,
                'height': h,
                'world_view_transform': world_view_transform,
                'full_proj_transform': full_proj_transform,
                'camera_center': camera_center
            }
            ret[f'novel_view{i}'] = novel_view_data

        if cfg.save_video:
            rendering_video_meta = []
            rendering_video_meta0 = []
            render_path_mode = 'interpolate'            
            poses_paths = self.get_video_rendering_path(ref_poses=src_exts, mode=render_path_mode, near_far=None, train_c2w_all=None, n_frames=60)
            for pose in poses_paths[0]:
                R = np.array(pose[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
                T = np.array(pose[:3, 3], np.float32)
                for i in range(cfg.credsplatting.cas_config.num):
                    h, w = H*cfg.credsplatting.cas_config.render_scale[i], W*cfg.credsplatting.cas_config.render_scale[i]
                    tar_ixt_ = tar_ixt.copy()
                    tar_ixt_[:2,:] *= cfg.credsplatting.cas_config.render_scale[i]
                    FovX = data_utils.focal2fov(tar_ixt[0, 0], w)
                    FovY = data_utils.focal2fov(tar_ixt[1, 1], h)
                    projection_matrix = data_utils.getProjectionMatrix(znear=self.znear, zfar=self.zfar, K=tar_ixt, h=H, w=W).transpose(0, 1)
                    world_view_transform = torch.tensor(data_utils.getWorld2View2(R, T, np.array(self.trans), self.scale)).transpose(0, 1)
                    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                    camera_center = world_view_transform.inverse()[3, :3]
                    if i==0:
                        rendering_meta0 = {
                            'FovX':  torch.FloatTensor([FovX]),
                            'FovY':  torch.FloatTensor([FovY]),
                            'width': w,
                            'height': h,
                            'world_view_transform': world_view_transform,
                            'full_proj_transform': full_proj_transform,
                            'camera_center': camera_center
                        }
                    else:
                        rendering_meta = {
                            'FovX':  torch.FloatTensor([FovX]),
                            'FovY':  torch.FloatTensor([FovY]),
                            'width': W,
                            'height': H,
                            'world_view_transform': world_view_transform,
                            'full_proj_transform': full_proj_transform,
                            'camera_center': camera_center,
                            'tar_ext': pose
                        }
                for i in range(cfg.credsplatting.cas_config.num):
                    tar_ext[:3] = pose
                    rays, _, _ = credsplatting_utils.build_rays(tar_img, tar_ext, tar_ixt, tar_mask, i, self.split)
                    rendering_meta.update({f'rays_{i}': rays})
                rendering_video_meta.append(rendering_meta)
                rendering_video_meta0.append(rendering_meta0)
            ret['rendering_video_meta1'] = rendering_video_meta
            ret['rendering_video_meta0'] = rendering_video_meta0
        return ret
    
    def get_video_rendering_path(self, ref_poses, mode, near_far, train_c2w_all, n_frames=60, batch=None):
        # loop over batch
        poses_paths = []
        ref_poses = ref_poses[None]
        for batch_idx, cur_src_poses in enumerate(ref_poses):
            if mode == 'interpolate':
                # convert to c2ws
                pose_square = torch.eye(4).unsqueeze(0).repeat(cur_src_poses.shape[0], 1, 1)
                cur_src_poses = torch.from_numpy(cur_src_poses)
                pose_square[:, :3, :] = cur_src_poses[:,:3]
                cur_c2ws = pose_square.double().inverse()[:, :3, :].to(torch.float32).cpu().detach().numpy()
                cur_path = get_interpolate_render_path(cur_c2ws, n_frames)
            elif mode == 'spiral':
                cur_c2ws_all = train_c2w_all
                cur_near_far = near_far.tolist()
                rads_scale = 0.3
                cur_path = get_spiral_render_path(cur_c2ws_all, cur_near_far, rads_scale=rads_scale, N_views=n_frames)
            else:
                raise Exception(f'Unknown video rendering path mode {mode}')

            # convert back to extrinsics tensor
            cur_w2cs = torch.tensor(cur_path).inverse()[:, :3].to(torch.float32)
            poses_paths.append(cur_w2cs)

        poses_paths = torch.stack(poses_paths, dim=0)
        return poses_paths

    def read_src(self, scene_info, src_views):
        inps, exts, ixts = [], [], []
        for i, src_view in enumerate(src_views):
            # if self.split == 'train':
            #     if random.random() < 0.03:
            #         random_number = random.randint(1, 6)
            #         load_src_image_path = scene_info['img_paths'][src_view].replace('_3_', f'_{random_number}_')
            #     else:
            load_src_image_path = scene_info['img_paths'][src_view]
            image_np = (np.array(imageio.imread(load_src_image_path)) / 255.) * 2. - 1.
            # image_np = self.center_image(imageio.imread(load_src_image_path), 'mean')

            ext = scene_info['exts'][src_view]
            ixt = scene_info['ixts'][src_view]
            orig_size = image_np.shape[:2][::-1]
            # print(orig_size)
            image_np = cv2.resize(image_np, self.input_h_w[::-1], interpolation=cv2.INTER_AREA)
            ixt[0] *= self.input_h_w[1] / orig_size[0]
            ixt[1] *= self.input_h_w[0] / orig_size[1]
            # print(image_np.shape)
            # inps.append((np.array(imageio.imread(scene_info['img_paths'][src_view])) / 255.))
            # if i == 0:
            #     image_np = self.crop_center(image_np, 0.1, 0.5)
            # if i==1:
            #     image_np = self.crop_center(image_np, 0.1, 0.2)
            # if i==2:
            #     image_np = self.crop_center(image_np, 0.1, 0.8)
            # if self.split == 'train':
            #     if random.random() < 0.05:
            #         image_np = self.crop_box(image_np)
            #     if random.random() < 0.05:
            #         image_np = self.add_mask(image_np)
            #     if random.random() < 0.05:
            #         image_np = self.add_gaussian_noise(image_np)

            inps.append(image_np)
            exts.append(ext)
            ixts.append(ixt)
        return np.stack(inps).transpose((0, 3, 1, 2)).astype(np.float32), np.stack(exts), np.stack(ixts)

    def __len__(self):
        return len(self.metas)

    def crop_center(self, img, facter, position_facter):
        target_height, target_width, _ = img.shape

        # 计算矩形实际尺寸
        block_width = int(target_width * facter)
        block_height = int(target_height * facter)

        # 计算随机位置（确保矩形在图像内部）
        x_start = int(target_width * position_facter)
        y_start = int(target_height * position_facter)
        # print(x_start, y_start, target_width, target_height)
        # 创建掩码，挖掉矩形区域（设为黑色）
        # mask = np.zeros((target_height, target_width, 1), dtype=np.float32)
        img[y_start-block_height:y_start+block_height, x_start-block_width:x_start+block_width, :] = -1.0

        return img

    def crop_box(self, img):
        target_height, target_width, _ = img.shape
        # 生成随机矩形区域的参数
        # 矩形宽度占比 (0.1-0.4之间)
        block_width_ratio = np.random.uniform(0.05, 0.25)
        # 矩形高度占比 (0.1-0.4之间)
        block_height_ratio = np.random.uniform(0.05, 0.25)

        # 计算矩形实际尺寸
        block_width = int(target_width * block_width_ratio)
        block_height = int(target_height * block_height_ratio)

        # 计算随机位置（确保矩形在图像内部）
        x_start = np.random.randint(0, target_width - block_width)
        y_start = np.random.randint(0, target_height - block_height)

        # 创建掩码，挖掉矩形区域（设为黑色）
        # mask = np.zeros((target_height, target_width, 1), dtype=np.float32)
        # if random.random():
        img[y_start:y_start+block_height, x_start:x_start+block_width, :] = -1.0
        # else:
        #     img[y_start:y_start+block_height, x_start:x_start+block_width, :] = 1.0

        return img


    def add_mask(self, img):
        height, width, _ = img.shape
        mask_ratio = 0.1
        mask = np.random.choice([0, 1], size=(height, width, 1), p=[1-mask_ratio, mask_ratio])

        # 应用掩码，将需要挖掉的区域设为黑色（值为-1）
        img = img * (1 - mask) + (-1) * mask

        return img

    def add_gaussian_noise(self, image, mean=0, var=0.001):
        """向图像添加高斯噪声"""
        sigma = var **0.5
        # 生成与图像同形状的高斯噪声
        gauss = np.random.normal(mean, sigma, image.shape)
        # 添加噪声并确保值在[-1, 1]范围内
        noisy_image = image + gauss
        noisy_image = np.clip(noisy_image, -1, 1)
        return noisy_image

    def center_image(self, img, mode='mean'):
        # scale 0~255 to 0~1
        # np_img = np.array(img, dtype=np.float32) / 255.
        # return np_img
        # normalize image input
        if mode == 'standard':
            np_img = np.array(img, dtype=np.float32) / 255.
        elif mode == 'mean':
            img_array = np.array(img)
            img = img_array.astype(np.float32)
            # img = img.astype(np.float32)
            var = np.var(img, axis=(0, 1), keepdims=True)
            mean = np.mean(img, axis=(0, 1), keepdims=True)
            return (img - mean) / (np.sqrt(var) + 0.00000001)
        else:
            raise Exception("{}? Not implemented yet!".format(mode))

        return np_img