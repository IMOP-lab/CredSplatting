from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network, load_pretrain
from lib.evaluators import make_evaluator
import torch.multiprocessing
import torch
import torch.distributed as dist
import os
import numpy as np
import random
import logging

if cfg.fix_random:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_file_path):
    # seed = 42 + dist.get_rank()
    # # print(seed)
    # torch.manual_seed(seed) #pytorch随机seed
    # np.random.seed(seed)    #numpy随机seed
    # random.seed(seed)   #python随机seed

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    # 确保日志文件的目录存在
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 创建一个文件处理器，并设置级别为DEBUG
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 创建一个控制台处理器，并设置级别为WARNING
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 获取根logger，并设置级别为DEBUG
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 设置Pillow库的日志级别为WARNING
    logging.getLogger('PIL').setLevel(logging.WARNING)

def train(cfg, network):
    # 指定日志文件路径
    log_file_path = os.path.join(cfg.trained_model_dir,'./logging.log')
    setup_logging(log_file_path)

    print(cfg.distributed)
    train_loader = make_data_loader(cfg,
                                    is_train=True,
                                    is_distributed=cfg.distributed,
                                    max_iter=cfg.ep_iter)
    if cfg.skip_eval:
        val_loader = None
    else:
        val_loader = make_data_loader(cfg, is_train=False)
    trainer = make_trainer(cfg, network, train_loader)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)

    begin_epoch = load_model(network,
                             optimizer,
                             scheduler,
                             recorder,
                             cfg.trained_model_dir,
                             resume=cfg.resume)
    if begin_epoch == 0 and cfg.pretrain != '':
        load_pretrain(network, cfg.pretrain)

    set_lr_scheduler(cfg, scheduler)
    psnr_best, ssim_best, lpips_best = 0, 0, 10
    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        if cfg.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)
        train_loader.dataset.epoch = epoch

        # print(111111)
        trainer.train(epoch, train_loader, optimizer, recorder)
        # print(22222)
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0 and cfg.local_rank == 0:
            save_model(network, optimizer, scheduler, recorder,
                       cfg.trained_model_dir, epoch)
            
        if (epoch + 1) % cfg.save_latest_ep == 0 and cfg.local_rank == 0:
            save_model(network,
                       optimizer,
                       scheduler,
                       recorder,
                       cfg.trained_model_dir,
                       epoch,
                       last=True)

        if not cfg.skip_eval and (epoch + 1) % cfg.eval_ep == 0 and cfg.local_rank == 0:
            result = trainer.val(epoch, val_loader, evaluator, recorder)
            psnr = result['psnr']
            ssim = result['ssim']
            lpips = result['lpips']
            if psnr > psnr_best:
                psnr_best = psnr
                save_model(network,
                       optimizer,
                       scheduler,
                       recorder,
                       cfg.trained_model_dir,
                       epoch,
                       custom='psnr_best')
            if ssim > ssim_best:
                ssim_best = ssim
                save_model(network,
                       optimizer,
                       scheduler,
                       recorder,
                       cfg.trained_model_dir,
                       epoch,
                       custom='ssim_best')
            if lpips < lpips_best:
                lpips_best = lpips
                save_model(network,
                       optimizer,
                       scheduler,
                       recorder,
                       cfg.trained_model_dir,
                       epoch,
                       custom='lpips_best')
            print(f'psnr_best: {psnr_best:.2f}, ssim_best: {ssim_best:.3f}, lpips_best: {lpips_best:.3f}')

        if not cfg.skip_eval and (epoch + 1) % cfg.eval_ep == 0 and cfg.local_rank > 0:
            trainer.val(epoch, val_loader, evaluator, recorder)


    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(network,
                         cfg.trained_model_dir,
                         resume=cfg.resume,
                         epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def main():
    # print(cfg.distributed)
    if cfg.distributed:
        # 检查并设置默认环境变量（适用于单节点调试）
        # if not os.environ.get("RANK"):
        #     os.environ["RANK"] = "0"
        # if not os.environ.get("WORLD_SIZE"):
        #     os.environ["WORLD_SIZE"] = "1"
        # if not os.environ.get("MASTER_ADDR"):
        #     os.environ["MASTER_ADDR"] = "127.0.0.1"
        # if not os.environ.get("MASTER_PORT"):
        #     os.environ["MASTER_PORT"] = "29500"
        # cfg.local_rank = int(os.environ['RANK']) % torch.cuda.device_count()
        cfg.local_rank = int(os.environ.get('RANK', 0)) % torch.cuda.device_count()
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    # if cfg.get('require_view_selection'):
    #     view_selection_file = os.path.join(cfg.result_dir, f'view_selection.json')
    #     if not os.path.exists(view_selection_file):
    #         print("\033[93mView selection file not found. Preprocessing...\033[0m")
    #         run_preprocess()

    torch.autograd.set_detect_anomaly(False)

    network = make_network(cfg)
    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)
    if cfg.local_rank == 0:
        print('Success!')
        print('='*80)
    os.system('kill -9 {}'.format(os.getpid()))


if __name__ == "__main__":
    main()
