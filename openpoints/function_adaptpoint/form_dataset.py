#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/8 20:43
# @Author  : wangjie

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

#####################################
# data loader with two output
#####################################
class Form_dataset_cls(Dataset):
    def __init__(self, pointcloud, label, idx, x=None, unmasked=None, origin_x=None):
        assert pointcloud is not None
        self.pointcloud = np.concatenate(pointcloud)
        self.label = np.concatenate(label)
        self.idx = np.concatenate(idx)
        if x is not None:
            self.x = np.concatenate(x)
        if unmasked is not None:
            self.unmasked = np.concatenate(unmasked)
        if origin_x is not None:
            self.origin_x = np.concatenate(origin_x)
        assert self.pointcloud.shape[0] == self.label.shape[0]

    def __getitem__(self, item):
        out_pointcloud = self.pointcloud[item]
        out_label = self.label[item]
        out_idx = self.idx[item]
        x_ = self.x[item]
        unmasked_ = self.unmasked[item]
        origin_x_ = self.origin_x[item]
        data = {'pos': out_pointcloud,
                'y': out_label,
                'idx': out_idx,
                'x': x_,
                'unmasked_pos': unmasked_,
                'origin_x': origin_x_
                }
        # out_pointcloud = torch.from_numpy(out_pointcloud).float()
        # out_label = torch.from_numpy(out_label).int()

        return data

    def __len__(self):
        return self.pointcloud.shape[0]


class Form_dataset_shapenet(Dataset):
    def __init__(self, pos, y, heights, cls):
        assert pos is not None
        self.pos = np.concatenate(pos)
        self.y = np.concatenate(y)
        self.heights = np.concatenate(heights)
        self.cls = np.concatenate(cls)

        assert self.pos.shape[0] == self.y.shape[0]

    def __getitem__(self, item):
        pos_ = self.pos[item]
        y_ = self.y[item]
        heights_ = self.heights[item]
        cls_ = self.cls[item]
        data = {'pos': pos_,
                'y': y_,
                'heights': heights_,
                'cls': cls_,
                }
        # out_pointcloud = torch.from_numpy(out_pointcloud).float()
        # out_label = torch.from_numpy(out_label).int()

        return data

    def __len__(self):
        return self.pos.shape[0]