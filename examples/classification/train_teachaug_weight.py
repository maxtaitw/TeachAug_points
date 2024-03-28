#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/8 16:35
# @Author  : wangjie

import os, logging, csv, numpy as np, wandb
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
# from torch.optim.swa_utils import get_ema_multi_avg_fn
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious, PCA
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
# from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample, fps
from examples.classification.train_pointwolf_utils import train_one_epoch_pointwolf, train_one_epoch_rsmix
from openpoints.models_adaptpoint import build_adaptpointmodels_from_cfg
from openpoints.function_adaptpoint import Form_dataset_cls, get_feedback_loss_ver1, get_feedback_loss_teacher, get_feedback_loss_teacher_v2, get_feedback_loss_teacher_weight
from openpoints.online_aug.pointwolf import PointWOLF_classversion
import h5py
from openpoints.utils import Summary
from openpoints.dataset.scanobjectnn_c.scanobjectnn_c import ScanObjectNNC, eval_corrupt_wrapper_scanobjectnnc
from openpoints.loss import build_criterion_from_cfg

import matplotlib.pyplot as plt
from tsnecuda import TSNE

def copyfiles(cfg):
    import shutil
    #   copy pointcloud model
    path_copy = f'{cfg.run_dir}/copyfile'
    if not os.path.isdir(path_copy):
        os.makedirs(path_copy)
    shutil.copy(f'{os.path.realpath(__file__)}', path_copy)
    shutil.copytree('openpoints', f'{path_copy}/openpoints')
    pass

def write_to_csv(oa, macc, accs, best_epoch, cfg, write_header=True):
    accs_table = [f'{item:.2f}' for item in accs]
    header = ['method', 'OA', 'mAcc'] + \
        cfg.classes + ['best_epoch', 'log_path', 'wandb link']
    data = [cfg.exp_name, f'{oa:.3f}', f'{macc:.2f}'] + accs_table + [
        str(best_epoch), cfg.run_dir, wandb.run.get_url() if cfg.wandb.use_wandb else '-']
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()

def print_cls_results(oa, macc, accs, epoch, cfg):
    s = f'\nClasses\tAcc\n'
    for name, acc_tmp in zip(cfg.classes, accs):
        s += '{:10}: {:3.2f}%\n'.format(name, acc_tmp)
    s += f'E@{epoch}\tOA: {oa:3.2f}\tmAcc: {macc:3.2f}\n'
    logging.info(s)

def save_augmenter(generator, path, epoch):
    state = {
        'generator': generator.state_dict(),
    }
    path = path + '/augmenter'
    if not os.path.isdir(path):
        os.makedirs(path)
    filepath = os.path.join(path, f"augmenter_{epoch}.pth")
    torch.save(state, filepath)

def get_gan_model(cfg):
    """
    return PointAug augmentor and discriminator
    and corresponding optimizer and scheduler
    """
    # Create model: G and D
    print("==> Creating model...")
    # generator
    generator = build_adaptpointmodels_from_cfg(cfg.adaptmodel_gan).cuda()
    # generator = PointWOLF_classversion4().to(device)
    print("==> Total parameters of Generator: {:.2f}M"\
          .format(sum(p.numel() for p in generator.parameters()) / 1000000.0))

    if cfg.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        # discriminator = nn.parallel.DistributedDataParallel(
        #     discriminator.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)


    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfg.adaptpoint_params.lr_generator, betas=(cfg.adaptpoint_params.b1, cfg.adaptpoint_params.b2))

    criterion_gan = torch.nn.BCELoss()
    dict = {
        'model_G': generator,
        'optimizer_G': optimizer_G,
        'criterion_gan': criterion_gan
    }
    return dict

def train_gan(cfg, gan_dict, train_loader, summary, writer, epoch, model_student, model_teacher, loss_init=None):
    generator = gan_dict['model_G']
    optimizer_G = gan_dict['optimizer_G']
    criterion_gan = gan_dict['criterion_gan']
    generator.train()
    model_student.eval()
    # model_teacher.eval()
    # prepare buffer list for update
    tmp_out_buffer_list = []
    tmp_points_buffer_list = []
    tmp_label_buffer_list = []
    tmp_idx_buffer_list = []
    tmp_unmasked_buffer_list = []
    tmp_originx_buffer_list = []
    aug_count = 0
    # pointwolf = PointWOLF_classversion(**cfg.pointwolf)
    for i, data in tqdm(enumerate(train_loader), total=train_loader.__len__()):
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        if 'origin_x' in data and data['origin_x'] is not None:
            origin_x = data['origin_x']
        else:
            origin_x = data['x'].clone()
        points = data['x']
        label = data['y']
        idx = data['idx']
        if 'unmasked_pos' in data and data['unmasked_pos'] is not None:
            points[:, :, :3] = data['unmasked_pos']
        points_clone = points.clone()
        # points_unmasked = points.clone()
        input_pointcloud = points[:, :, :3].contiguous()

        #  Train Generator
        _, unmasked_pos, gen_imgs, sample_weight = generator(input_pointcloud)
        # g_loss_raw = criterion_gan(discriminator(gen_imgs), real_label)

        points[:, :, :3] = gen_imgs

        data_fake = {
            'pos': points[:, :, :3].contiguous(),
            'y': label,
            'x': points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous(),
        }
        data_real = {
            'pos': origin_x[:, :, :3].contiguous(),
            'y': label,
            'x': origin_x[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous(),
        }
        # data_prev = {
        #     'pos': points_clone[:, :, :3].contiguous(),
        #     'y': label,
        #     'x': points_clone[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous(),
        # }

        # feedback_loss_ratio = cfg.get('feedbackloss_ratio', 1)

        feedback_loss, pred_fake_stu, pred_fake_tea = get_feedback_loss_teacher_weight(cfg=cfg, model_student=model_student, model_teacher=model_teacher, \
                                            data_real=data_real, data_fake=data_fake, sample_weight=sample_weight, \
                                            epoch=epoch, summary=summary, writer=writer)
            

        # print(f"gard before backward: {generator.predict_prob_layer.embedding.net[0].weight.grad}")
        optimizer_G.zero_grad()
        feedback_loss.backward(torch.ones_like(feedback_loss))

        optimizer_G.step()
        summary.summary_train_iter_num_update()

        if epoch > 0:
            pred_fake_tea = torch.nn.functional.softmax(pred_fake_tea, dim=1)
            max_scores_tea, max_idx_tea = torch.max(pred_fake_tea, dim=1)
            pred_fake_stu = torch.nn.functional.softmax(pred_fake_stu, dim=1)
            max_scores_stu, max_idx_stu = torch.max(pred_fake_stu, dim=1)
            print('label', label)
            print('teacher', max_idx_tea)
            print('student', max_idx_stu)
            # print(pred_fake_tea)
            # max_mean = torch.mean(max_scores_tea)

            # threshold = get_threshold(cfg, epoch, loss_init_s=loss_init_s)
            # threshold = loss_init_s*0.5
            # writer.add_scalar('train_G_iter/max_mean', max_mean, summary.train_iter_num)
            # print(loss_init_s, threshold)
            out_points, unmasked_pos, count = mask_data(pred_fake_stu, max_scores_tea, max_idx_tea, points, unmasked_pos, origin_x, label, epoch)
            aug_count += count
        else:
            out_points = points

        tmp_out_buffer_list.append(out_points[:, :, :3].detach().cpu().numpy())
        tmp_label_buffer_list.append(label.detach().cpu().numpy())
        tmp_idx_buffer_list.append(idx.detach().cpu().numpy())
        tmp_points_buffer_list.append(out_points.detach().cpu().numpy())
        tmp_unmasked_buffer_list.append(unmasked_pos.detach().cpu().numpy())
        tmp_originx_buffer_list.append(origin_x.detach().cpu().numpy())

        # print(f'{i}-th, g_loss:{g_loss}, d_loss:{d_loss}')
    # buffer loader will be used to save fake pose pair
    print('\nprepare buffer loader for train on fake pose')
    # print(aug_count)
    writer.add_scalar('train_G_iter/aug_count', aug_count, summary.train_iter_num)
    model_student.zero_grad()
    # model_teacher.zero_grad()
    # if ((epoch%25)<5):
    #     save_augmenter(generator=generator, path=cfg.run_dir, epoch=epoch)
    fake_dataset = Form_dataset_cls(tmp_out_buffer_list, tmp_label_buffer_list, tmp_idx_buffer_list, tmp_points_buffer_list, tmp_unmasked_buffer_list, tmp_originx_buffer_list)
    weight = sample_weight.detach().cpu().numpy()
    return fake_dataset, weight

def mask_data(pred_fake_stu, max_scores_tea, max_idx_tea, points, unmasked_pos, origin_x, label, epoch):
    # Find the maximum prediction score along the class dimension
    pred_fake_stu = torch.nn.functional.softmax(pred_fake_stu, dim=1)
    # max_score_stu, max_idx_stu = torch.max(pred_fake_stu, dim=1)
    
    score_stu = torch.gather(pred_fake_stu, 1, max_idx_tea.unsqueeze(1))
    score_stu = torch.flatten(score_stu)
    # max_scores, _ = torch.max(pred_fake_stu, dim=-1)
    # print(label)
    # print(score_stu)
    th = get_threshold(epoch, max_scores_tea)

    # mask_idx = max_idx_stu == max_idx_tea
    # print(mask_idx.sum())
    
    # Create a mask based on whether the maximum score is smaller than the threshold
    # mask = score_stu > (max_scores_tea*0.5)
    mask = score_stu > th
    
    # Expand the mask dimensions to match the size of data
    mask = mask.unsqueeze(-1).unsqueeze(-1)
    count = mask.sum()
    print(count)
    
    # Mask out the data A using the mask and replace it with data B
    points = torch.where(mask, points, origin_x)
    unmasked_pos = torch.where(mask, unmasked_pos, origin_x[:, :, :3])
    
    return points, unmasked_pos, count

def get_threshold(epoch, max_scores_tea=1):
    # ------------------------------------DASH------------------------------------
    # Selection Stage: 
    # if epoch > cfg.num_warmup_epoch:#+1000:
    #     t = epoch - cfg.num_warmup_epoch + 1
    if epoch > 0:
        t = epoch + 1

        C = 1.0001
        gama = 1.005
        mu = t - 1
        threshold_s = C * pow(gama, -mu) * max_scores_tea
        threshold_s = torch.clamp(threshold_s, 0.5)
    else:
        threshold_s = 500.0
    return threshold_s


def main(gpu, cfg, profile=False):
    copyfiles(cfg)
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0 :
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        # writer = SummaryWriter(log_dir=cfg.run_dir)
        summary = Summary(cfg.run_dir)
        writer = summary.create_summary()
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if not cfg.model.get('criterion_args', False):
        cfg.model.criterion_args = cfg.criterion_args
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    # criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    ######################################################
    # pretrained teacher
    if cfg.pretrained_teacher.use_pretrained == True:
        teacher_model = build_model_from_cfg(cfg.teacher_model).cuda()
        logging.info('loading pretrained teacher model ...')
        teacher_state_dict = torch.load(cfg.pretrained_teacher.pretrained_teacher_path, map_location='cpu')
        teacher_model.load_state_dict(teacher_state_dict['model'])
    # EMA Teacher
    else:
        avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
                    cfg.ema_rate * averaged_model_parameter + (1 - cfg.ema_rate) * model_parameter
        teacher_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=avg_fn)
        teacher_model.train()
        for ema_p in teacher_model.parameters():
            ema_p.requires_grad_(False)


    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    test_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                            cfg.dataset,
                                            cfg.dataloader,
                                            datatransforms_cfg=cfg.datatransforms,
                                            split='test',
                                            distributed=cfg.distributed
                                            )
    num_classes = val_loader.dataset.num_classes if hasattr(
        val_loader.dataset, 'num_classes') else None
    num_points = val_loader.dataset.num_points if hasattr(
        val_loader.dataset, 'num_points') else None
    if num_classes is not None:
        assert cfg.num_classes == num_classes
    logging.info(f"number of classes of the dataset: {num_classes}, "
                 f"number of points sampled from dataset: {num_points}, "
                 f"number of points as model input: {cfg.num_points}")
    cfg.classes = cfg.get('classes', None) or val_loader.dataset.classes if hasattr(
        val_loader.dataset, 'classes') else None or np.range(num_classes)
    validate_fn = eval(cfg.get('val_fn', 'validate'))

    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler,
                              pretrained_path=cfg.pretrained_path)
            macc, oa, accs, cm = validate_fn(model, val_loader, cfg)
            print_cls_results(oa, macc, accs, cfg.start_epoch, cfg)
        else:
            if cfg.mode == 'test':
                # test mode
                epoch, best_val = load_checkpoint(
                    model, pretrained_path=cfg.pretrained_path)
                macc, oa, accs, cm = validate_fn(model, test_loader, cfg)
                print_cls_results(oa, macc, accs, epoch, cfg)
                return True
            elif cfg.mode == 'val':
                # validation mode
                epoch, best_val = load_checkpoint(model, cfg.pretrained_path)
                macc, oa, accs, cm = validate_fn(model, val_loader, cfg)
                print_cls_results(oa, macc, accs, epoch, cfg)
                return True
            elif cfg.mode == 'finetune':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model.encoder, cfg.pretrained_path)
    else:
        logging.info('Training from scratch')
    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    origin_train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    
    path_PCA = f'{cfg.run_dir}/PCA'
    if not os.path.isdir(path_PCA):
        os.makedirs(path_PCA)
    # ===> start training
    val_macc, val_oa, val_accs, best_val, macc_when_best, best_epoch = 0., 0., [], 0., 0., 0
    model.zero_grad()
    gan_model_dict = get_gan_model(cfg)
    
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1

        if epoch > cfg.get('num_warmup_epoch', 0):
            fake_dataset, sample_weight = train_gan(cfg, gan_model_dict, train_loader, summary, writer, epoch, model, teacher_model, loss_init=None)
            fake_train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                                     cfg.dataset,
                                                     cfg.dataloader,
                                                     datatransforms_cfg=cfg.datatransforms,
                                                     split='train',
                                                     distributed=cfg.distributed,
                                                     dataset=fake_dataset,
                                                     )
            print('train fake')
            train_loss, train_macc, train_oa, _, _ = \
                train_one_epoch(model, teacher_model, fake_train_loader, sample_weight,
                            optimizer, scheduler, epoch, cfg)
            
        else:
            sample_weight = None
            train_loss, train_macc, train_oa, _, _ = \
                train_one_epoch(model, teacher_model, train_loader, sample_weight,
                                optimizer, scheduler, epoch, cfg)

        # reset train loader
        # if (epoch+1) > int(cfg.continued_start_epoch):
        #     train_loader = fake_train_loader
        #     if (epoch+1) % int(cfg.continued_cycle) == 0 or (epoch+1)<50:
        #         train_loader = origin_train_loader

        is_best = False
        if epoch % cfg.val_freq == 0:
            val_macc, val_oa, val_accs, val_cm = validate_fn(
                model, val_loader, cfg)
            tea_val_macc, tea_val_oa, tea_val_accs, _ = validate_fn(
                teacher_model, val_loader, cfg)
            is_best = val_oa > best_val
            if is_best:
                best_val = val_oa
                macc_when_best = val_macc
                best_epoch = epoch
                logging.info(f'Find a better ckpt @E{epoch}')
            print_cls_results(val_oa, val_macc, val_accs, epoch, cfg)
            print_cls_results(tea_val_oa, tea_val_macc, tea_val_accs, epoch, cfg)

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_oa {train_oa:.2f}, val_oa {val_oa:.2f}, best val oa {best_val:.2f}')
        if writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_oa', train_macc, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('val_oa', val_oa, epoch)
            writer.add_scalar('mAcc_when_best', macc_when_best, epoch)
            writer.add_scalar('best_val', best_val, epoch)
            writer.add_scalar('epoch', epoch, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best
                            )
        
        
    # test the last epoch
    test_macc, test_oa, test_accs, test_cm = validate(model, test_loader, cfg)
    print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)
    if writer is not None:
        writer.add_scalar('test_oa', test_oa, epoch)
        writer.add_scalar('test_macc', test_macc, epoch)

    # test the best validataion model
    best_epoch, _ = load_checkpoint(model, pretrained_path=os.path.join(
        cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    test_macc, test_oa, test_accs, test_cm = validate(model, test_loader, cfg)
    if writer is not None:
        writer.add_scalar('test_oa', test_oa, best_epoch)
        writer.add_scalar('test_macc', test_macc, best_epoch)
    print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)

    best_ckpt_path = os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth')
    last_ckpt_path = os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_latest.pth')
    # testscanobjectnnc(model=model, path=best_ckpt_path, cfg=cfg)
    # testscanobjectnnc(model=model, path=last_ckpt_path, cfg=cfg)

    

    if writer is not None:
        writer.close()
    if cfg.distributed:
        dist.destroy_process_group()

def get_analyze_data(model, train_loader, epoch, cfg):
    model.eval()
    # feature PCA
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    features = None
    targets = None
    indices = None
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        points = data['x']
        target = data['y']
        index = data['idx']

        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        feat = model.get_feat(data)
        
        filter = torch.where((target == 2) | (target == 5) | (target == 8))
        filtered_feat = feat[filter]
        filtered_target = target[filter]
        filtered_index = index[filter]

        current_feature = filtered_feat.detach().cpu().numpy()
        current_target = filtered_target.detach().cpu().numpy()
        current_index = filtered_index.detach().cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_feature))
            targets = np.concatenate((targets, current_target))
            indices = np.concatenate((indices, current_index))
        else:
            features = current_feature
            targets = current_target
            indices = current_index
        
        analyze_data = [features, targets, indices]
    
    return analyze_data

def tsne_viz(analyze_data, prev_analyze_data, epoch, cfg):
    # Feature analyze
    features, targets, indices = analyze_data
    # sorted_indices = np.argsort(indices)
    # sorted_features = features[sorted_indices]
    # sorted_targets = targets[sorted_indices]

    if prev_analyze_data is not None and (epoch%5 != 0):
        prev_features, prev_targets, prev_indices = prev_analyze_data
        concatenated_features = np.concatenate((prev_features, features), axis=0)
        tsne = TSNE(n_components=2, perplexity=50).fit_transform(concatenated_features)
        prev_tsne = tsne[0:len(prev_features), ]
        current_tsne = tsne[len(prev_features):, ]
        plt.scatter(prev_tsne[:, 0], prev_tsne[:, 1], s=3, c=prev_targets, alpha=0.5, cmap='viridis', marker='^')
        plt.scatter(current_tsne[:, 0], current_tsne[:, 1], s=5, c=targets, cmap='plasma')

    else: 
        current_tsne = TSNE(n_components=2, perplexity=50).fit_transform(features)
        # print('tsne', tsne.shape)
        plt.scatter(current_tsne[:, 0], current_tsne[:, 1], s=5, c=targets)
    
    plt.savefig(f'{cfg.run_dir}/PCA/{int(epoch)}.png')
    plt.clf()
 

def train_one_epoch(model, teacher_model, train_loader, sample_weight, optimizer, scheduler, epoch, cfg):
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points

    model.train()  # set model to training mode
    teacher_model.train()

    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        # print(data.keys())
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        points = data['x']
        target = data['y']
        index = data['idx']
        """ bebug
        from openpoints.dataset import vis_points
        vis_points(data['pos'].cpu().numpy()[0])
        """
        num_curr_pts = points.shape[1]
        if num_curr_pts > npoints:  # point resampling strategy
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
            if  points.size(1) < point_all:
                point_all = points.size(1)
            fps_idx = furthest_point_sample(
                points[:, :, :3].contiguous(), point_all)
            fps_idx = fps_idx[:, np.random.choice(
                point_all, npoints, False)]
            points = torch.gather(
                points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1]))

        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        if cfg.use_sample_weight == True:
            logits, loss = model.get_logits_weighted_loss(data, target, sample_weight) if not hasattr(model, 'module') else model.module.get_logits_weighted_loss(data, target, sample_weight)
        else:
            logits, loss = model.get_logits_loss(data, target) if not hasattr(model, 'module') else model.module.get_logits_loss(data, target) 

        loss.backward()


        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            model.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        if cfg.pretrained_teacher.use_pretrained is False:
            # print('update')
            teacher_model.update_parameters(model)
            # print(list(teacher_model.parameters()))
            
        # update confusion matrix
        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item())
        if idx % cfg.print_freq == 0:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")

    macc, overallacc, accs = cm.all_acc()
    return loss_meter.avg, macc, overallacc, accs, cm



@torch.no_grad()
def validate(model, val_loader, cfg):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
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
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(count)
    macc, overallacc, accs = cm.cal_acc(tp, count)
    return macc, overallacc, accs, cm

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
    return {'acc': (overallacc/100)}

def testscanobjectnnc(model, path, cfg):
    ckpt = torch.load(f'{path}')
    model.load_state_dict(ckpt['model'])
    epoch  = ckpt['epoch']
    eval_corrupt_wrapper_scanobjectnnc(model, validate_scanobjectnnc, {'cfg': cfg},
                               cfg.run_dir, epoch)