# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import argparse
import datetime
import logging
import math
import os
import random
import sys
import time
import numpy as np
import wandb

from os import path as osp
# root_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录（basicsr）
# sys.path.append(os.path.dirname(root_dir))  # 将根目录（NAFNet）加入路径
# 增强版路径配置：明确获取项目根目录并添加到搜索路径
# 1. 获取当前脚本（train.py）的绝对路径
current_script_path = os.path.abspath(__file__)
# 2. 获取当前脚本所在目录（basicsr目录）
basicsr_dir = os.path.dirname(current_script_path)
# 3. 获取项目根目录（basicsr的上一级目录，即Res_FFT/Res_FFT）
root_dir = os.path.dirname(basicsr_dir)
# 4. 强制将根目录添加到Python搜索路径（如果已存在则先移除再添加，确保优先级）
if root_dir in sys.path:
    sys.path.remove(root_dir)
sys.path.insert(0, root_dir)  # 插入到最前面，优先搜索

# 验证路径是否正确（可选，用于调试）
print(f"当前脚本路径: {current_script_path}")
print(f"basicsr目录路径: {basicsr_dir}")
print(f"项目根目录路径: {root_dir}")
print(f"Python搜索路径: {sys.path[:3]}")  # 打印前3个路径确认


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str, init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse
import math
from basicsr.data.transforms import (  # 直接从transforms.py导入
    paired_random_crop,
    paired_random_crop_hw,
    augment,
    mod_crop,
    img_rotate,
)
import basicsr.data.transforms as transforms

#重新初始化 cuDNN
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--input_path', type=str, required=False, help='The path to the input image. For single image inference only.')
    parser.add_argument('--output_path', type=str, required=False, help='The path to the output image. For single image inference only.')

    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    if args.input_path is not None and args.output_path is not None:
        opt['img_path'] = {
            'input_img': args.input_path,
            'output_img': args.output_path
        }

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    # if (opt['logger'].get('wandb')
    #         is not None) and (opt['logger']['wandb'].get('project')
    #                           is not None) and ('debug' not in opt['name']):
    #     assert opt['logger'].get('use_tb_logger') is True, (
    #         'should turn on tensorboard when using wandb')
    #     init_wandb_logger(opt)
        # 将配置文件参数上传到WandB
        # wandb.config.update(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        # tb_logger = init_tb_logger(log_dir=f'./logs/{opt['name']}') #mkdir logs @CLY
        tb_logger = init_tb_logger(log_dir=osp.join('logs', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # automatic resume ..
    state_folder_path = 'experiments/{}/training_states/'.format(opt['name'])
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    resume_state = None
    if len(states) > 0:
        print('!!!!!! resume state .. ', states, state_folder_path)
        max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
        resume_state = os.path.join(state_folder_path, max_state_file)
        opt['path']['resume_state'] = resume_state

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()
    best_psnr = -float('inf') #best_psnr = -36.9
    # for epoch in range(start_epoch, total_epochs + 1):
    epoch = start_epoch
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data, is_val=False)
            model.optimize_parameters(current_iter, tb_logger)

            result_code = model.optimize_parameters(current_iter, tb_logger)
            # if result_code == -1 and tb_logger:
            #     print('loss explode .. ')
            #     exit(0)
            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                # print('msg logger .. ', current_iter)
                msg_logger(log_vars)
                 # 手动将训练指标同步到WandB（确保实时性）
                #wandb.log({k: v for k, v in log_vars.items() if k not in ['epoch', 'iter', 'total_iter']}, step=current_iter)
            # 保存最好的loss
            # if model.log_dict['l_pix'] < best_psnr:
            #     psnr = model.log_dict['l_pix']
            #     logger.info(f'********** best models:{current_iter} loss:{psnr} ***********')
            #     model.save(epoch, current_iter)
            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
            # if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                # wheather use uint8 image to compute metrics
                use_image = opt['val'].get('use_image', True)
                val_metrics = model.validation(val_loader, current_iter, tb_logger,
                                 opt['val']['save_img'], rgb2bgr, use_image )
                # 将验证指标同步到WandB
                # if val_metrics:
                #     wandb.log({f'val_{k}': v for k, v in val_metrics.items()}, step=current_iter)
                #     # 记录最佳PSNR
                #     if val_metrics.get('psnr', -float('inf')) > best_psnr:
                #         best_psnr = val_metrics['psnr']
                #         wandb.log({'best_val_psnr': best_psnr}, step=current_iter)
                #         logger.info(f'Update best PSNR: {best_psnr:.4f} at iter {current_iter}')
                log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)


            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter
        epoch += 1

    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        rgb2bgr = opt['val'].get('rgb2bgr', True)
        use_image = opt['val'].get('use_image', True)
        metric = model.validation(val_loader, current_iter, tb_logger,
                         opt['val']['save_img'], rgb2bgr, use_image)
        if metric:
            wandb.log({f'final_val_{k}': v for k, v in final_metrics.items()}, step=current_iter)
        # if tb_logger:
        #     print('xxresult! ', opt['name'], ' ', metric)
    if tb_logger:
        tb_logger.close()
     # 结束WandB运行
    # if opt['rank'] == 0 and wandb.run is not None:
    #     wandb.finish()


if __name__ == '__main__':
    import os
    os.environ['GRPC_POLL_STRATEGY']='epoll1'
    main()
