
from . import samplers
import torch
import torch.utils.data
import importlib.util
import os
from .collate_batch import make_collator
import numpy as np
import time
from lib.config.config import cfg
from torch.utils.data import DataLoader
import cv2
cv2.setNumThreads(1)

from torch.utils.data import ConcatDataset as TorchConcatDataset
from torch.utils.data import Dataset

class ConcatDatasetWithTupleIndex(TorchConcatDataset):
    """
    支持元组索引的ConcatDataset子类
    适配需要 (数据索引, input_views_num) 双输入的数据集，仅解析元组中的数据索引用于子数据集定位
    """
    def __getitem__(self, index_meta):
        # 从元组中提取真实的"数据索引"（元组第0位），忽略input_views_num（元组第1位）
        data_index, input_views_num = index_meta
        # print(11111111111111111111)
        # 1. 找到该data_index对应的子数据集（与原生ConcatDataset逻辑一致）
        dataset_idx = 0
        # 遍历子数据集的累积长度，定位目标子数据集
        for i, dataset in enumerate(self.datasets):
            if data_index < len(dataset):
                dataset_idx = i
                break
            data_index -= len(dataset)  # 减去前一个子数据集的长度，得到在当前子数据集内的索引
        
        # 2. 用 (子数据集内索引, input_views_num) 调用子数据集的__getitem__
        # 保持与子数据集要求的输入格式一致（元组索引）
        return self.datasets[dataset_idx][(data_index, input_views_num)]

    def __len__(self):
        # 保持与原生ConcatDataset一致的长度（所有子数据集长度之和）
        return super().__len__()
    
def make_dataset(cfg, is_train=True):
    if is_train:
        dataset_configs = cfg.train_dataset
        # 全局默认模块键（供未指定模块的数据集使用）
        default_module_key = "train_dataset_module"
        default_path_key = "train_dataset_path"
    else:
        dataset_configs = cfg.test_dataset
        default_module_key = "test_dataset_module"
        default_path_key = "test_dataset_path"

    if isinstance(dataset_configs, dict):
        dataset_configs = [dataset_configs]
    
    datasets = []
    dataset_weights = []
    dataset_sizes = []

    for idx, ds_cfg in enumerate(dataset_configs):
        # 关键：优先使用数据集自身的模块配置，否则用全局默认
        # （每个数据集可指定独立的 module 和 path，对应不同处理代码）
        ds_module = ds_cfg.get("module", cfg.get(default_module_key))  # 数据集专属模块名
        ds_path = ds_cfg.get("path", cfg.get(default_path_key))        # 数据集专属.py路径
        
        # 加载当前数据集的专属模块（不同数据集加载不同.py文件）
        if not os.path.exists(ds_path):
            raise FileNotFoundError(f"数据集{idx+1}的处理文件不存在：{ds_path}")
        spec = importlib.util.spec_from_file_location(ds_module, ds_path)
        ds_module_obj = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ds_module_obj)
        
        # 加载当前数据集的专属Dataset类（需确保每个.py中都有"Dataset"类，或在配置中指定类名）
        # 若不同数据集的类名不同，可在配置中加"dataset_class"字段（如 ds_class_name = ds_cfg.get("dataset_class", "Dataset")）
        if not hasattr(ds_module_obj, "Dataset"):
            raise AttributeError(
                f"数据集{idx+1}的处理文件 {ds_path} 中未定义 'Dataset' 类，请检查类名是否正确"
            )
        DatasetClass = ds_module_obj.Dataset

        # 初始化当前数据集（传入该数据集的专属参数）
        dataset = DatasetClass(**ds_cfg)
        ds_name = ds_cfg.get("name", f"dataset_{idx+1}")
        ds_size = len(dataset)
        ds_weight = ds_cfg.get("sample_weight", 1.0)

        # 记录信息（便于调试）
        print(f"加载{('训练' if is_train else '测试')}数据集: {ds_name}")
        print(f"  - 处理文件: {ds_path}")
        print(f"  - 样本数: {ds_size}, 采样权重: {ds_weight}")
        datasets.append(dataset)
        dataset_weights.append(ds_weight)
        dataset_sizes.append(ds_size)

    # 合并数据集并附加信息
    if len(datasets) == 1:
        concat_dataset = datasets[0]
        concat_dataset.dataset_info = (dataset_sizes, dataset_weights)
    else:
        concat_dataset = ConcatDatasetWithTupleIndex(datasets)
        concat_dataset.dataset_info = (dataset_sizes, dataset_weights)
        print(f"合并{len(datasets)}个数据集，总样本数: {sum(dataset_sizes)}")
    
    return concat_dataset


# 以下函数（make_data_sampler/make_batch_data_sampler等）完全不变，复用权重平衡逻辑
def make_data_sampler(dataset, shuffle, is_distributed, is_train):
    if is_distributed:
        # print(12345)
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if is_train and hasattr(dataset, "dataset_info"):
        dataset_sizes, dataset_weights = dataset.dataset_info
        num_datasets = len(dataset_sizes)
        if num_datasets > 1 and not all(w == 1.0 for w in dataset_weights):
            print(f"启用数据集加权采样，权重配置: {dataset_weights}")
            print(dataset_weights, dataset_sizes)
            weight_coeffs = [w / s for w, s in zip(dataset_weights, dataset_sizes)]
            sample_weights = [dataset_weights, dataset_sizes]
            for coeff, size in zip(weight_coeffs, dataset_sizes):
                sample_weights.extend([coeff] * size)
            total_samples = sum(dataset_sizes)
            return torch.utils.data.sampler.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=total_samples,
                replacement=False
            )
    
    if shuffle:
        return torch.utils.data.sampler.RandomSampler(dataset)
    else:
        return torch.utils.data.sampler.SequentialSampler(dataset)


def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter, is_train):
    if is_train:
        batch_sampler = cfg.train.batch_sampler
        sampler_meta = cfg.train.sampler_meta
    else:
        batch_sampler = cfg.test.batch_sampler
        sampler_meta = cfg.test.sampler_meta
    
    if batch_sampler == 'default':
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last)
    elif batch_sampler == 'image_size':
        batch_sampler = samplers.ImageSizeBatchSampler(sampler, batch_size,
                                                       drop_last, sampler_meta)
    elif batch_sampler == 'credsplatting':
        batch_sampler = samplers.credsplattingBatchSampler(
            sampler, batch_size, drop_last, sampler_meta, is_train)
    
    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, max_iter)
    return batch_sampler


def worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def make_data_loader(cfg, is_train=True, is_distributed=False, max_iter=-1):
    if is_train:
        batch_size = cfg.train.batch_size
        shuffle = cfg.train.shuffle
        drop_last = False
    else:
        batch_size = cfg.test.batch_size
        shuffle = True if is_distributed else False
        drop_last = False

    dataset = make_dataset(cfg, is_train)
    sampler = make_data_sampler(dataset, shuffle, is_distributed, is_train)
    batch_sampler = make_batch_data_sampler(
        cfg, sampler, batch_size, drop_last, max_iter, is_train)
    
    num_workers = cfg.train.num_workers
    collator = make_collator(cfg, is_train)
    data_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collator,
        worker_init_fn=worker_init_fn,
        # pin_memory=True
    )

    return data_loader
    