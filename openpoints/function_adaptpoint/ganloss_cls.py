#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/8 21:02
# @Author  : wangjie
import logging
import torch
import torch.nn.functional as F
import numpy as np
from ..loss import build_criterion_from_cfg, DistillLoss
from ..loss import LabelSmoothingCrossEntropy as CE
from ..loss import SWD

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss_raw = -(one_hot * log_prb).sum(dim=1)
        loss = loss_raw.mean()

    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss, loss_raw

# def get_feedback_loss_ver2_1(cfg, model_pointcloud, data_real, data_fake, summary, writer):
def get_feedback_loss_ver1(cfg, model_pointcloud, data_real, data_fake, epoch, summary, writer):
    '''
    To generate harder case
    '''
    def update_hardratio(start, end, current_epoch, total_epoch):
        return start + (end - start) * current_epoch / total_epoch

    def fix_hard_ratio_loss(expected_hard_ratio, harder, easier):  # similar to MSE
        fix_loss = torch.abs(1 - torch.exp(harder - expected_hard_ratio * easier))
        return fix_loss.mean()
        # return torch.abs(1 - torch.exp(harder - expected_hard_ratio * easier))

    #   get loss on real/fake data
    model_pointcloud.eval()
    pred_fake = model_pointcloud.forward(data_fake)                     #   [B, 40]
    pred_real = model_pointcloud.forward(data_real)                     #   [B, 40]
    label = data_real['y']
    criterion = build_criterion_from_cfg(cfg.criterion_args)
    # loss_fake, loss_raw_fake = criterion(pred_fake, label.long())       #   loss_fake: [1]   loss_raw_fake: [B]
    # loss_real, loss_raw_real = criterion(pred_real, label.long())       #   loss_real: [1]   loss_raw_real: [B]
    loss_fake = criterion(pred_fake, label.long())       #   loss_fake: [1]   loss_raw_fake: [B]
    loss_real = criterion(pred_real, label.long())       #   loss_real: [1]   loss_raw_real: [B]

    #   updata hardratio
    hardratio = update_hardratio(cfg.adaptpoint_params.hardratio_s, cfg.adaptpoint_params.hardratio, epoch, cfg.epochs)

    # feedback_loss = fix_hard_ratio_loss(hardratio, loss_raw_fake, loss_raw_real)
    feedback_loss = fix_hard_ratio_loss(hardratio, loss_fake, loss_real)

    writer.add_scalar('train_G_iter/loss_fakedata', loss_fake.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/loss_realdata', loss_real.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/loss_ratio', loss_fake.item()/loss_real.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/hardratio', hardratio, summary.train_iter_num)
    return feedback_loss


def get_feedback_loss_teacher(cfg, model_pointcloud, model_teacher, data_real, data_fake, epoch, summary, writer):
    '''
    To generate harder case
    '''
    def update_hardratio(start, end, current_epoch, total_epoch):
        return start + (end - start) * current_epoch / total_epoch

    def fix_hard_ratio_loss(expected_hard_ratio, harder, easier):  # similar to MSE
        fix_loss = torch.abs(1 - torch.exp(harder - expected_hard_ratio * easier))
        return fix_loss.mean()
        # return torch.abs(1 - torch.exp(harder - expected_hard_ratio * easier))

    #   get loss on real/fake data
    model_pointcloud.eval()
    pred_fake = model_pointcloud.forward(data_fake)                     #   [B, 40]
    pred_real = model_pointcloud.forward(data_real)                     #   [B, 40]
    label = data_real['y']
    criterion = build_criterion_from_cfg(cfg.criterion_args)
    # loss_fake, loss_raw_fake = criterion(pred_fake, label.long())       #   loss_fake: [1]   loss_raw_fake: [B]
    # loss_real, loss_raw_real = criterion(pred_real, label.long())       #   loss_real: [1]   loss_raw_real: [B]
    loss_fake = criterion(pred_fake, label.long())       #   loss_fake: [1]   loss_raw_fake: [B]
    loss_real = criterion(pred_real, label.long())       #   loss_real: [1]   loss_raw_real: [B]

    #   teacher loss
    model_teacher.eval()
    pred_fake_tea = model_teacher.forward(data_fake)                     #   [B, 40]
    pred_real_tea = model_teacher.forward(data_real)                     #   [B, 40]
    # label = data_real['y']
    criterion_tea = build_criterion_from_cfg(cfg.teacher_criterion_args)
    # loss_fake, loss_raw_fake = criterion(pred_fake, label.long())       #   loss_fake: [1]   loss_raw_fake: [B]
    # loss_real, loss_raw_real = criterion(pred_real, label.long())       #   loss_real: [1]   loss_raw_real: [B]
    loss_fake_tea = criterion_tea(pred_fake_tea, label.long())       #   loss_fake: [1]   loss_raw_fake: [B]
    # loss_real_tea = criterion_tea(pred_real_tea, label.long())       #   loss_real: [1]   loss_raw_real: [B]


    #   updata hardratio
    hardratio = update_hardratio(cfg.adaptpoint_params.hardratio_s, cfg.adaptpoint_params.hardratio, epoch, cfg.epochs)

    # feedback_loss = fix_hard_ratio_loss(hardratio, loss_raw_fake, loss_raw_real)
    feedback_loss = fix_hard_ratio_loss(hardratio, loss_fake, loss_real) + loss_fake_tea

    writer.add_scalar('train_G_iter/loss_fakedata', loss_fake.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/loss_realdata', loss_real.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/loss_ratio', loss_fake.item()/loss_real.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/loss_fake_teacher', loss_fake_tea.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/hardratio', hardratio, summary.train_iter_num)
    return feedback_loss


def get_feedback_loss_teacher_v2(cfg, model_pointcloud, model_teacher, data_real, data_fake, epoch, summary, writer):
    '''
    To generate harder case
    '''
    def update_hardratio(start, end, current_epoch, total_epoch):
        return start + (end - start) * current_epoch / total_epoch

    def fix_hard_ratio_loss(expected_hard_ratio, harder, easier):  # similar to MSE
        fix_loss = torch.abs(1 - torch.exp(harder - expected_hard_ratio * easier))
        return fix_loss.mean()
        # return torch.abs(1 - torch.exp(harder - expected_hard_ratio * easier))

    #   get loss on real/fake data
    model_pointcloud.eval()
    pred_fake = model_pointcloud.forward(data_fake)                     #   [B, 40]
    label = data_real['y']
    criterion_stu = build_criterion_from_cfg(cfg.teacher_criterion_args)
    loss_fake_stu = criterion_stu(pred_fake, label.long())       #   loss_fake: [1]   loss_raw_fake: [B]

    #   teacher loss
    model_teacher.eval()
    pred_fake_tea = model_teacher.forward(data_fake)                     #   [B, 40]
    # pred_real_tea = model_teacher.forward(data_real)                     #   [B, 40]
    criterion_tea = build_criterion_from_cfg(cfg.teacher_criterion_args)
    loss_fake_tea = criterion_tea(pred_fake_tea, label.long())       #   loss_fake: [1]   loss_raw_fake: [B]
    # loss_real_tea = criterion_tea(pred_real_tea, label.long())       #   loss_fake: [1]   loss_raw_fake: [B]

    #   updata hardratio
    hardratio = update_hardratio(cfg.adaptpoint_params.hardratio_s, cfg.adaptpoint_params.hardratio, epoch, cfg.epochs)

    # feedback_loss = fix_hard_ratio_loss(hardratio, loss_raw_fake, loss_raw_real)
    teacher_loss = fix_hard_ratio_loss(hardratio, loss_fake_stu, loss_fake_tea)
    # f_loss = fix_hard_ratio_loss(hardratio, loss_fake, loss_real)

    #   EMD
    real_points = data_real['x']
    fake_points = data_fake['x']
    # print(f'original_point_num{len(real_points)}, augmented_point_num{len(fake_points)}')
    SWD_criterion = SWD(num_projs=100)
    SWD_loss_dict = SWD_criterion(real_points, fake_points)
    swd_loss = SWD_loss_dict['loss'].mean(dim=0)

    w_swd = cfg.loss_weight.w_swd
    w_tea = cfg.loss_weight.w_tea
    feedback_loss = w_swd*swd_loss + w_tea*teacher_loss
    # feedback_loss = loss_fake_stu + d_loss

    writer.add_scalar('train_G_iter/loss_fake_stu', loss_fake_stu.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/loss_fake_tea', loss_fake_tea.item(), summary.train_iter_num)
    # writer.add_scalar('train_G_iter/loss_real_tea', loss_real_tea.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/feedback_loss', feedback_loss.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/swd_loss', swd_loss.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/teacher_loss', teacher_loss.item(), summary.train_iter_num)
    return feedback_loss

def get_feedback_loss_teacher_weight(cfg, model_pointcloud, model_teacher, data_real, data_fake, sample_weight, epoch, summary, writer):
    '''
    To generate harder case
    '''
    def update_hardratio(start, end, current_epoch, total_epoch):
        return start + (end - start) * current_epoch / total_epoch

    def fix_hard_ratio_loss(expected_hard_ratio, harder, easier):  # similar to MSE
        fix_loss = torch.abs(1 - torch.exp(harder - expected_hard_ratio * easier))
        return fix_loss
        # return torch.abs(1 - torch.exp(harder - expected_hard_ratio * easier))
    # sample_weight = sample_weight.cpu().detach().numpy()
    # print(sample_weight.shape)
    # print(sample_weight)
    #   get loss on real/fake data
    model_pointcloud.eval()
    pred_fake = model_pointcloud.forward(data_fake)                     #   [B, 40]
    label = data_real['y']
    criterion_stu = build_criterion_from_cfg(cfg.weight_teacher_criterion_args)
    # criterion_stu.__init__(sample_weight=sample_weight)
    loss_fake_stu = criterion_stu(pred_fake, label.long())       #   loss_fake: [1]   loss_raw_fake: [B]

    #   teacher loss
    model_teacher.eval()
    pred_fake_tea = model_teacher.forward(data_fake)                     #   [B, 40]
    # pred_real_tea = model_teacher.forward(data_real)                     #   [B, 40]
    criterion_tea = build_criterion_from_cfg(cfg.weight_teacher_criterion_args)
    # criterion_tea.__init__(sample_weight=sample_weight)
    loss_fake_tea = criterion_tea(pred_fake_tea, label.long())       #   loss_fake: [1]   loss_raw_fake: [B]
    # loss_real_tea = criterion_tea(pred_real_tea, label.long())       #   loss_fake: [1]   loss_raw_fake: [B]

    #   updata hardratio
    hardratio = update_hardratio(cfg.adaptpoint_params.hardratio_s, cfg.adaptpoint_params.hardratio, epoch, cfg.epochs)

    # feedback_loss = fix_hard_ratio_loss(hardratio, loss_raw_fake, loss_raw_real)
    teacher_loss = fix_hard_ratio_loss(hardratio, loss_fake_stu, loss_fake_tea)
    # f_loss = fix_hard_ratio_loss(hardratio, loss_fake, loss_real)

    #   EMD
    real_points = data_real['x']
    fake_points = data_fake['x']
    # print(f'original_point_num{len(real_points)}, augmented_point_num{len(fake_points)}')
    SWD_criterion = SWD(num_projs=100)
    SWD_loss_dict = SWD_criterion(real_points, fake_points)
    swd_loss = SWD_loss_dict['loss']

    norm_sample_weight = sample_weight.squeeze() / sample_weight.sum()
    print(norm_sample_weight)
    w_swd = cfg.loss_weight.w_swd
    w_tea = cfg.loss_weight.w_tea
    weighted_swd = (w_swd * swd_loss * norm_sample_weight).sum().unsqueeze(0)
    weighted_teacher = (w_tea * teacher_loss * norm_sample_weight).sum().unsqueeze(0)
    feedback_loss = weighted_swd + weighted_teacher
    # print(sample_weight.shape)
    # print(swd_loss.shape)
    # print(weighted_swd.shape)
    # print(teacher_loss.shape)
    # print(weighted_teacher.shape)
    # print(feedback_loss.shape)
    # feedback_loss = loss_fake_stu + d_loss

    writer.add_scalar('train_G_iter/loss_fake_stu', loss_fake_stu.mean().item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/loss_fake_tea', loss_fake_tea.mean().item(), summary.train_iter_num)
    # writer.add_scalar('train_G_iter/loss_real_tea', loss_real_tea.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/feedback_loss', feedback_loss.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/swd_loss', weighted_swd.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/teacher_loss', weighted_teacher.item(), summary.train_iter_num)
    return feedback_loss

def get_feedback_loss_teacher_distill(cfg, model_pointcloud, model_teacher, data_real, data_fake, epoch, summary, writer):
    '''
    To generate harder case
    '''
    def update_hardratio(start, end, current_epoch, total_epoch):
        return start + (end - start) * current_epoch / total_epoch

    def fix_hard_ratio_loss(expected_hard_ratio, harder, easier):  # similar to MSE
        fix_loss = torch.abs(1 - torch.exp(harder - expected_hard_ratio * easier))
        return fix_loss.mean()
        # return torch.abs(1 - torch.exp(harder - expected_hard_ratio * easier))

    #   get loss on real/fake data
    model_pointcloud.eval()
    pred_fake = model_pointcloud.forward(data_fake)                     #   [B, 40]
    label = data_real['y']
    distill_loss_fn = DistillLoss(base_criterion_args=cfg.distill_args)
    distill_loss = distill_loss_fn(data_fake, pred_fake, label.long(), model_teacher)       #   loss_fake: [1]   loss_raw_fake: [B]

    #   teacher loss
    model_teacher.eval()
    pred_fake_tea = model_teacher.forward(data_fake)                     #   [B, 40]
    criterion_tea = build_criterion_from_cfg(cfg.teacher_criterion_args)
    loss_fake_tea = criterion_tea(pred_fake_tea, label.long())       #   loss_fake: [1]   loss_raw_fake: [B]


    feedback_loss = distill_loss + loss_fake_tea

    writer.add_scalar('train_G_iter/distill_loss', distill_loss.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/loss_fake_tea', loss_fake_tea.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/feedback_loss', feedback_loss.item(), summary.train_iter_num)

    return feedback_loss