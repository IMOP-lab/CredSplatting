import numpy as np
import matplotlib.pyplot as plt
import os

def save_feature_heatmap(feature, save_path, cmap='jet', dpi=300, flag=False):
    """
    将输入的特征图转换为热图并保存到指定路径。

    参数:
        feature (numpy.ndarray): 输入的特征图，形状为 (w, h)。
        save_path (str): 保存热图的文件路径（包括文件名和扩展名）。
        cmap (str): 热图的颜色映射，默认为 'viridis'。
        dpi (int): 保存图像的分辨率，默认为 300。

    返回:
        None
    """
    print(np.max(feature), np.min(feature), feature.shape)
    # 检查输入是否为二维数组
    if not isinstance(feature, np.ndarray) or feature.ndim != 2:
        raise ValueError("输入的特征图必须是形状为 (w, h) 的二维 numpy 数组。")
    eps = 1e-8
    feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature) + eps)
    if flag:
        # feature.clip()
        feature = feature *  8.5
        feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))

    # 创建热图
    plt.figure(figsize=(4, 3))  # 设置图像大小
    w, h = feature.shape
    plt.imshow(feature, cmap=cmap)  # 使用指定的颜色映射绘制热图
    cbar = plt.colorbar()  # 添加颜色条
    # cbar.set_clim(0, 1)  # 显式设置颜色条范围为 [0, 1]
    plt.axis('off')  # 关闭坐标轴

    # 确保保存路径的目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存热图
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭图像以释放内存

import os
import matplotlib.pyplot as plt

def save_feature_heatmap4(feature, save_path, cmap='jet', dpi=300, flag=False):
    """
    保存热图，确保输出图像大小与 feature 的像素尺寸一致。
    
    参数：
    - feature: 输入的二维数组 (w, h)，表示热图数据。
    - cmap: 颜色映射（如 'viridis', 'jet', 等）。
    - save_path: 保存路径。
    - dpi: 分辨率，默认为 100。
    """
    # 获取 feature 的形状
    w, h = feature.shape

    print(np.max(feature), np.min(feature), feature.shape, 111)
    # 检查输入是否为二维数组
    if not isinstance(feature, np.ndarray) or feature.ndim != 2:
        raise ValueError("输入的特征图必须是形状为 (w, h) 的二维 numpy 数组。")
    eps = 1e-8
    feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature) + eps)
    if flag:
        # feature.clip()
        feature = feature *  8.5
        feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))

    # 计算 figsize，确保输出图像大小与 feature 的像素尺寸一致
    figsize = (h / dpi * 4, w / dpi * 4)

    # 创建热图
    plt.figure(figsize=figsize, dpi=dpi)  # 设置图像大小和分辨率
    plt.imshow(feature, cmap=cmap)  # 使用指定的颜色映射绘制热图
    # cbar = plt.colorbar()  # 添加颜色条
    plt.axis('off')  # 关闭坐标轴

    # 确保保存路径的目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存热图
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭图像以释放内存


from PIL import Image
import numpy as np
def save_feature_heatmap3(feature, save_path, cmap='jet', dpi=300, flag=False):

    # print(np.unique(feature))
    arr  = (feature * 0.5 + 0.5)
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8)

    imageio.imwrite(save_path, arr .astype(np.uint8))


import matplotlib.cm as cm
import imageio
def save_feature_heatmap2(feature, save_path, cmap='hsv', dpi=300):

    # print(np.max(feature), np.min(feature), feature.shape)
    # 检查输入是否为二维数组
    if not isinstance(feature, np.ndarray) or feature.ndim != 2:
        raise ValueError("输入的特征图必须是形状为 (w, h) 的二维 numpy 数组。")
    
    colormap = cm.get_cmap('jet')

    feature = colormap((feature - np.min(feature)) / (np.max(feature) - np.min(feature)))
    

    imageio.imwrite(save_path, (feature * 255.).astype(np.uint8))

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