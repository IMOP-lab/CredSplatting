import numpy as np
import matplotlib.pyplot as plt
import os

def save_feature_heatmap(feature, save_path, cmap='jet', dpi=300):
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
    
    # print(np.unique(feature))

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