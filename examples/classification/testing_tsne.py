#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/8 13:56
# @Author  : wangjie

import __init__
import os, argparse, yaml, numpy as np
from torch import multiprocessing as mp
from examples.classification.train import main as train
from openpoints.utils import EasyConfig, dist_utils, find_free_port, generate_exp_directory, resume_exp_directory, Wandb
from openpoints.utils import EasyConfig
from openpoints.models import build_model_from_cfg
from openpoints.dataset import build_dataloader_from_cfg,build_dataset_from_cfg
from openpoints.utils import registry
from openpoints.transforms import build_transforms_from_cfg
from openpoints.models_adaptpoint import build_adaptpointmodels_from_cfg

import torch
from easydict import EasyDict as edict
import datetime
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from tqdm import tqdm
# from openpoints.dataset.scanobjectnn_c.scanobjectnn_c import ScanObjectNNC, eval_corrupt_wrapper
import sklearn.metrics as metrics
import h5py
import matplotlib.pyplot as plt
from tsnecuda import TSNE
import re


def build_dataset_own(cfg=None, mode='val'):
    #   build datatransforms
    data_transform = build_transforms_from_cfg(mode, cfg.datatransforms)
    split_cfg = cfg.dataset.get(mode, edict())
    split_cfg.transform = data_transform
    dataset = build_dataset_from_cfg(cfg.dataset.common, split_cfg)
    return dataset



def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)



@torch.no_grad()
def validate_scanobjectnnc(split, model, cfg):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points
    data_transform = build_transforms_from_cfg('val', cfg.datatransforms_scanobjectnn_c)
    testloader = torch.utils.data.DataLoader(ScanObjectNNC(split=split, transform=data_transform), num_workers=int(cfg.dataloader.num_workers), \
                            batch_size=cfg.get('val_batch_size', cfg.batch_size), shuffle=False, drop_last=False)
    pbar = tqdm(enumerate(testloader), total=testloader.__len__())
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        points = data['x']
        points = points[:, :npoints]
        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        logits = model(data)
        cm.update(logits.argmax(dim=1), target)

    tp, count = cm.tp, cm.count
    # if cfg.distributed:
    #     dist.all_reduce(tp), dist.all_reduce(count)
    macc, overallacc, accs = cm.cal_acc(tp, count)
    return {'acc': overallacc}
    # return macc, overallacc, accs, cm

def get_analyze_data(generator, model, train_loader, epoch, cfg):
    model.eval()
    generator.eval()
    # feature PCA
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    features_real = None
    features_fake = None
    targets = None
    indices = None
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        points = data['x']
        origin_points = points.clone()
        target = data['y']
        index = data['idx']

        input_pointcloud = points[:, :, :3].contiguous()
        _, unmasked_pos, gen_imgs, sample_weight = generator(input_pointcloud)
        points[:, :, :3] = gen_imgs

        data_real = {
            'pos': origin_points[:, :, :3].contiguous(),
            'y': target,
            'x': origin_points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous(),
        }
        data_fake = {
            'pos': points[:, :, :3].contiguous(),
            'y': target,
            'x': points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous(),
        }

        
        feat_fake = model(data_fake)
        feat_real = model(data_real)
        
        
        filter = torch.where((target == 2) | (target == 5) | (target == 8))
        filtered_feat_fake = feat_fake[filter]
        filtered_feat_real = feat_real[filter]
        filtered_target = target[filter]
        filtered_index = index[filter]

        current_feature_fake = filtered_feat_fake.detach().cpu().numpy()
        current_feature_real = filtered_feat_real.detach().cpu().numpy()
        current_target = filtered_target.detach().cpu().numpy()
        current_index = filtered_index.detach().cpu().numpy()
        # print(current_feature_fake.shape)
        if features_fake is not None:
            features_fake = np.concatenate((features_fake, current_feature_fake))
            features_real = np.concatenate((features_real, current_feature_real))
            targets = np.concatenate((targets, current_target))
            indices = np.concatenate((indices, current_index))
        else:
            features_fake = current_feature_fake
            features_real = current_feature_real
            targets = current_target
            indices = current_index
        
        analyze_data = [features_fake, features_real, targets, indices]
    
    return analyze_data

def tsne_viz(analyze_data, prev_analyze_data, epoch, cfg, root_path):
    # Feature analyze
    features_fake, features_real, targets, indices = analyze_data
    x_min, x_max = -200, 200  # Example range for x-axis
    y_min, y_max = -200, 200 # Example range for y-axis
    if features_real is not None and (epoch%5 != 0):
        # prev_features, prev_targets, prev_indices = prev_analyze_data
        concatenated_features = np.concatenate((features_real, features_fake), axis=0)
        tsne = TSNE(n_components=2, perplexity=50).fit_transform(concatenated_features)
        real_tsne = tsne[0:len(features_real), ]
        fake_tsne = tsne[len(features_fake):, ]
        plt.scatter(fake_tsne[:, 0], fake_tsne[:, 1], s=5, c=targets, cmap='plasma')
        plt.scatter(real_tsne[:, 0], real_tsne[:, 1], s=3, c=targets, alpha=0.5, cmap='viridis', marker='^')
        
    else: 
        real_tsne = TSNE(n_components=2, perplexity=50).fit_transform(features_fake)
        # print('tsne', tsne.shape)
        plt.scatter(real_tsne[:, 0], real_tsne[:, 1], s=5, c=targets)
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig(f'{root_path}/PCA/{int(epoch)}.png')
    plt.clf()


def find_checkpoint_file(folder_path, keyword='ckpt_best.pth'):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if keyword in file:
                return os.path.join(root, file)
    return None

def sort_by_number(filename):
    return int(filename.split('_')[1].split('.')[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser('S3DIS scene segmentation training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--profile', action='store_true',
                        default=False, help='set to True to profile speed')
    parser.add_argument('--msg_ckpt', default='ngpus1-seed6687-20230208-113827-C5JvDVLy53nEazUYXPtNyj', type=str, help='message after checkpoint')
    parser.add_argument('--testingmode', default='val', type=str, help='mode')
    parser.add_argument('--root_path', default=None, type=str, help='message after checkpoint')

    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    cfg.msg_ckpt = args.msg_ckpt
    cfg.testingmode = args.testingmode
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    torch.cuda.empty_cache()
    #   loading model
    model = build_model_from_cfg(cfg.model).cuda()
    # print(model)
    generator = build_adaptpointmodels_from_cfg(cfg.adaptmodel_gan).cuda()

    #   loading ckpt dir
    root_path = args.root_path
    checkpoint_path = f'{root_path}/checkpoint'
    ckpt_file_path = find_checkpoint_file(checkpoint_path)

    #   loading state dict
    state_dict = torch.load(ckpt_file_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])

    #   loading dataset
    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=False,
                                             )
    # test_macc, test_oa, test_accs, test_cm = validate(model, dataloader, cfg)
    # print(f'test oa: {test_oa}')
    aug_path = f"{root_path}/augmenter"
    PCA_path = f"{root_path}/PCA"
    if not os.path.isdir(PCA_path):
        os.makedirs(PCA_path)
    pattern = r'augmenter_(\d+)\.pth'
    prev_analyze_data = None

    # Get a list of file names in the folder and sort it based on the number part
    file_names = sorted(os.listdir(aug_path), key=sort_by_number)
    for aug_ckpt in file_names:
        match = re.match(pattern, aug_ckpt)
        epoch = int(match.group(1))
        print(epoch)

        aug_file = os.path.join(aug_path, aug_ckpt)
        aug_state_dict = torch.load(aug_file, map_location='cpu')
        generator.load_state_dict(aug_state_dict['generator'])
        analyze_data = get_analyze_data(generator, model, train_loader, epoch, cfg)
        tsne_viz(analyze_data, prev_analyze_data, epoch, cfg, root_path)
        prev_analyze_data = analyze_data.copy()
        

    # print(' ==> starting testing on scanobjectnnc ...')
    # # eval_corrupt_wrapper(model, validate_scanobjectnnc, {'cfg': cfg}, cfg.run_dir)
    # print(' ==> endinging testing on scanobjectnnc ...')

