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

if cfg.fix_random:
    random.seed(0)
    np.random.seed(0)

class Dataset:
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = os.path.join(cfg.workspace, kwargs['data_root'])
        self.split = kwargs['split']
        if 'scene' in kwargs:
            self.scenes = [kwargs['scene']]
        else:
            self.scenes = []
        self.build_metas(kwargs['ann_file'])
        self.depth_ranges = [425., 905.]
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = [0.0, 0.0, 0.0]
        self.scale = 1.0

    def build_metas(self, ann_file):
        scenes = [line.strip() for line in open(ann_file).readlines()]
        dtu_pairs = torch.load('data/credsplatting/pairs.th')

        self.scene_infos = {}
        self.metas = []
        if len(self.scenes) != 0:
            scenes = self.scenes

        for scene in scenes:
            scene_info = {'ixts': [], 'exts': [], 'dpt_paths': [], 'img_paths': []}
            for i in range(49):
                cam_path = os.path.join(self.data_root, 'Cameras/train/{:08d}_cam.txt'.format(i))
                ixt, ext, _ = data_utils.read_cam_file(cam_path)
                ext[:3, 3] = ext[:3, 3]
                ixt[:2] = ixt[:2] * 4
                dpt_path = os.path.join(self.data_root, 'Depths_raw/{}/depth_map_{:04d}.pfm'.format(scene, i))
                img_path = os.path.join(self.data_root, 'Rectified/{}_train/rect_{:03d}_3_r5000.png'.format(scene, i+1))
                scene_info['ixts'].append(ixt.astype(np.float32))
                scene_info['exts'].append(ext.astype(np.float32))
                scene_info['dpt_paths'].append(dpt_path)
                scene_info['img_paths'].append(img_path)

            if self.split == 'train' and len(self.scenes) != 1:
                train_ids = np.arange(49).tolist()
                test_ids = np.arange(49).tolist()
            elif self.split == 'train' and len(self.scenes) == 1:
                train_ids = dtu_pairs['dtu_train']
                test_ids = dtu_pairs['dtu_train']
            else:
                train_ids = dtu_pairs['dtu_train']
                test_ids = dtu_pairs['dtu_val']
            scene_info.update({'train_ids': train_ids, 'test_ids': test_ids})
            self.scene_infos[scene] = scene_info

            cam_points = np.array([np.linalg.inv(scene_info['exts'][i])[:3, 3] for i in train_ids])
            for tar_view in test_ids:
                cam_point = np.linalg.inv(scene_info['exts'][tar_view])[:3, 3]
                distance = np.linalg.norm(cam_points - cam_point[None], axis=-1)
                argsorts = distance.argsort()
                argsorts = argsorts[1:] if tar_view in train_ids else argsorts
                input_views_num = cfg.credsplatting.train_input_views[2] + 0 if self.split == 'train' else cfg.credsplatting.test_input_views
                src_views = [train_ids[i] for i in argsorts[:input_views_num]]
                self.metas += [(scene, tar_view, src_views)]

    def __getitem__(self, index_meta):
        index, input_views_num = index_meta
        scene, tar_view, src_views = self.metas[index]
        if self.split == 'train':
            if random.random() < 0.1:
                src_views = src_views + [tar_view]
            src_views = random.sample(src_views[:input_views_num+1], input_views_num)
        scene_info = self.scene_infos[scene]

        tar_img = np.array(imageio.imread(scene_info['img_paths'][tar_view])) / 255.
        H, W = tar_img.shape[:2]
        tar_ext, tar_ixt = scene_info['exts'][tar_view], scene_info['ixts'][tar_view]
        if self.split != 'train': # only used for evaluation
            tar_dpt = data_utils.read_pfm(scene_info['dpt_paths'][tar_view])[0].astype(np.float32)
            tar_dpt = cv2.resize(tar_dpt, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            tar_dpt = tar_dpt[44:556, 80:720]
            tar_mask = (tar_dpt > 0.).astype(np.uint8)
        else:
            tar_dpt = np.ones_like(tar_img)
            tar_mask = np.ones_like(tar_img)

        src_inps, src_exts, src_ixts = self.read_src(scene_info, src_views)


        if self.split == 'train' and random.random() < 0.3:
        # if random.random() < 0.3:
            # 1. 翻转目标视图
            # print(tar_img.shape)
            tar_img = self.flip_single_image(tar_img)
            # print(tar_ext, tar_ixt)
            tar_ext, tar_ixt = self.flip_single_camera(tar_ext, tar_ixt, H)
            # print(tar_ext, tar_ixt)
            # 翻转深度图（若有）
            tar_dpt = self.flip_single_image(tar_dpt)
            # 翻转mask（与图像一致）
            tar_mask = self.flip_single_image(tar_mask)

            # print(src_inps.shape, src_exts.shape, src_ixts.shape)
            # 2. 翻转源视图（批量处理）
            src_inps, src_exts, src_ixts = self.flip_batch_src_data(
                src_inps, src_exts, src_ixts, H
            )
            # print(src_inps.shape, src_exts.shape, src_ixts.shape)

        ret = {'src_inps': src_inps,
               'src_exts': src_exts,
               'src_ixts': src_ixts}
        ret.update({'tar_ext': tar_ext,
                    'tar_ixt': tar_ixt})
        # if self.split != 'train':
        ret.update({'tar_img': tar_img,
                    'tar_dpt': tar_dpt,
                    'tar_mask': tar_mask})
        ret.update({'near_far': np.array(self.depth_ranges).astype(np.float32)})
        ret.update({'meta': {'scene': scene, 'tar_view': tar_view, 'frame_id': 0}})

        for i in range(cfg.credsplatting.cas_config.num):
            rays, rgb, msk = credsplatting_utils.build_rays(tar_img, tar_ext, tar_ixt, tar_mask, i, self.split)
            s = cfg.credsplatting.cas_config.volume_scale[i]
            if self.split != 'train': # evaluation
                tar_dpt_i = cv2.resize(tar_dpt, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
                ret.update({f'tar_dpt_{i}': tar_dpt_i.astype(np.float32)})
            ret.update({f'rays_{i}': rays, f'rgb_{i}': rgb.astype(np.float32), f'msk_{i}': msk})
            ret['meta'].update({f'h_{i}': H, f'w_{i}': W})
            
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
            render_path_mode = 'interpolate'            
            poses_paths = self.get_video_rendering_path(ref_poses=src_exts, mode=render_path_mode, near_far=None, train_c2w_all=None, n_frames=60)
            for pose in poses_paths[0]:
                R = np.array(pose[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
                T = np.array(pose[:3, 3], np.float32)
                FovX = data_utils.focal2fov(tar_ixt[0, 0], W)
                FovY = data_utils.focal2fov(tar_ixt[1, 1], H)
                projection_matrix = data_utils.getProjectionMatrix(znear=self.znear, zfar=self.zfar, K=tar_ixt, h=H, w=W).transpose(0, 1)
                world_view_transform = torch.tensor(data_utils.getWorld2View2(R, T, np.array(self.trans), self.scale)).transpose(0, 1)
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]
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
            ret['rendering_video_meta'] = rendering_video_meta
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
            # image_np = (np.array(imageio.imread(load_src_image_path)) / 255.) * 2. - 1.
            image_np = (np.array(imageio.imread(load_src_image_path)))

            # image_np = self.center_image(imageio.imread(load_src_image_path), 'mean')

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
            #     if random.random() < 0.01:
            #         image_np = self.add_gaussian_noise(image_np)
            
            image_np = self.normalize(image_np / 255.)
            
            inps.append(image_np)
            exts.append(scene_info['exts'][src_view])
            ixts.append(scene_info['ixts'][src_view])
        return np.stack(inps).transpose((0, 3, 1, 2)).astype(np.float32), np.stack(exts), np.stack(ixts)


    def normalize(self, image):
        # print(image.shape)
        # 定义ImageNet的均值和标准差（RGB格式）
        imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
        # 归一化到0-1范围
        # image = image / 255.0
        # 应用ImageNet标准化
        return (image - imagenet_mean) / imagenet_std


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
        img[y_start-block_height:y_start+block_height, x_start-block_width:x_start+block_width, :] = 0.

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
        img[y_start:y_start+block_height, x_start:x_start+block_width, :] = -0.
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
    

    def center_image(self, img_array, mode='mean'):
        # scale 0~255 to 0~1
        # np_img = np.array(img, dtype=np.float32) / 255.
        # return np_img
        # normalize image input
        if mode == 'standard':
            np_img = np.array(img, dtype=np.float32) / 255.
        elif mode == 'mean':
            # img_array = np.array(img) / 255.
            img = img_array.astype(np.float32)
            # img = img.astype(np.float32)
            var = np.var(img, axis=(0, 1), keepdims=True)
            mean = np.mean(img, axis=(0, 1), keepdims=True)
            return (img - mean) / (np.sqrt(var) + 0.00000001)
        else:
            raise Exception("{}? Not implemented yet!".format(mode))

        return np_img


    def flip_single_image(self, img):
        """垂直翻转单张图像（H, W, 3）或深度图（H, W）"""
        return np.flipud(img).copy()

    def flip_single_camera(self, ext, ixt, H):
        """
        翻转单个相机的外参和内参（输出4×4齐次外参）
        Args:
            ext: 3x4 外参矩阵 [R | T]（输入仍保持3×4，兼容原有逻辑）
            ixt: 3x3 内参矩阵 [f_x, 0, c_x; 0, f_y, c_y; 0,0,1]
            H: 图像高度（用于计算新主点c_y）
        Returns:
            flipped_ext: 翻转后的4×4齐次外参矩阵
            flipped_ixt: 翻转后的3×3内参矩阵
        """
        # 1. 外参变换：绕x轴旋转180°
        rot_x_180 = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, 1]
        ], dtype=np.float32)

        flipped_R = rot_x_180 @ ext[:3, :3]          # 旋转更新
        flipped_T = rot_x_180 @ ext[:3, 3]           # 平移也用旋转矩阵左乘（关键修复！）

        # 2. 组合成3x4，再扩展为4x4
        flipped_ext_3x4 = np.hstack([flipped_R, flipped_T[:, None]])
        flipped_ext = np.eye(4, dtype=np.float32)
        flipped_ext[:3, :] = flipped_ext_3x4

        # 3. 内参：主点y坐标翻转
        flipped_ixt = ixt.copy()
        flipped_ixt[1, 2] = H - 1 - flipped_ixt[1, 2]  # cy' = H-1-cy

        return flipped_ext, flipped_ixt


    def flip_batch_src_data(self, src_inps, src_exts, src_ixts, H):
        """
        批量翻转源视图数据（src_inps是[N, C, H, W]格式）
        """
        # print(src_inps.shape)
        # 1. 源图像翻转：针对H维度（axis=3，因为src_inps是[N,C,H,W]）
        flipped_src_inps = np.flip(src_inps, axis=2).copy()

        # 2. 源外参/内参批量翻转
        flipped_src_exts = []
        flipped_src_ixts = []
        for ext, ixt in zip(src_exts, src_ixts):
            f_ext, f_ixt = self.flip_single_camera(ext, ixt, H)
            flipped_src_exts.append(f_ext)
            flipped_src_ixts.append(f_ixt)
        flipped_src_exts = np.stack(flipped_src_exts)
        flipped_src_ixts = np.stack(flipped_src_ixts)

        return flipped_src_inps, flipped_src_exts, flipped_src_ixts
    