#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/9 21:23
# @Author  : wangjie
import os
import glob
import h5py
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import argparse

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../fake_data')

red_rgb = np.array([255 / 255., 107 / 255., 107 / 255.])
green_rgb = np.array([107 / 255., 203 / 255., 119 / 255.])
blue_rgb = np.array([77 / 255, 150 / 255, 255 / 255])
purple = np.array([138 / 255, 163 / 255, 255 / 255])
red2green = red_rgb - green_rgb
green2blue = green_rgb - blue_rgb
red2blue = red_rgb - blue_rgb

classes = [
    "bag",
    "bin",
    "box",
    "cabinet",
    "chair",
    "desk",
    "display",
    "door",
    "shelf",
    "table",
    "bed",
    "pillow",
    "sink",
    "sofa",
    "toilet",
]

def o3dvis(points_1, points_2, points_3=None, id=None):
    '''
        points_1: coors of feat,[npoint, 3]
        points_2: coors of feat,[npoint, 3]
    '''
    width = 1080
    height = 1080

    front_size = [-0.078881283843093356, 0.98265290049739873, -0.16784224796908237]
    lookat_size = [0.057118464194894254, -0.010695673712330742, 0.047245129152854129]
    up_size = [0.018854223129214469, -0.16686616421723113, -0.98579927039414161]
    zoom = 0.98
    front = np.array(front_size)
    lookat = np.array(lookat_size)
    up = np.array(up_size)

    chessboard_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[0.1, 0.1, 0.1])


    color_1 = np.zeros_like(points_1)
    color_1[:] = red_rgb
    P_1 = o3d.geometry.PointCloud()
    P_1.points = o3d.utility.Vector3dVector(points_1)
    P_1.colors = o3d.utility.Vector3dVector(color_1)

    points_2 = points_2 - [1, 0, 0]
    color_2 = np.zeros_like(points_2)
    color_2[:] = blue_rgb
    # color_new[:] = green_rgb
    P_2 = o3d.geometry.PointCloud()
    P_2.points = o3d.utility.Vector3dVector(points_2)
    P_2.colors = o3d.utility.Vector3dVector(color_2)

    if points_3 is not None:
        points_3 = points_3 - [2, 0, 0]
        color_3 = np.zeros_like(points_3)
        color_3[:] = green_rgb
        # color_new[:] = green_rgb
        P_3 = o3d.geometry.PointCloud()
        P_3.points = o3d.utility.Vector3dVector(points_3)
        P_3.colors = o3d.utility.Vector3dVector(color_3)

    window_name = f"{id}"
    o3d.visualization.draw_geometries([P_1], window_name=window_name, width=width, height=height,
                                      zoom=zoom, front=front, lookat=lookat, up=up)
    


def load_h5_new(h5_name):
    f = h5py.File(h5_name, 'r')
    data_raw = f['raw'][:].astype('float32')
    data_raw_wpointwolf = f['raw_pointwolf'][:].astype('float32')
    data_fake = f['pointcloud'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    return data_raw, data_raw_wpointwolf, data_fake, label


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1, help="epoch")
    parser.add_argument("--minibatch", type=int, default=0, help="minibatch")
    parser.add_argument("--root_path", type=str, default=None, help="log_path")
    
    args = parser.parse_args()
    
    epoch = args.epoch
    minibatch = args.minibatch


    # path = f"/mnt/HDD8/max/TeachAugment_point/log/scanobjectnn/teachaugpoint_gen-teachaug_offset-20231122-191915-jAobx83NJMDaQnhMAkc2Ej/fakedata"
    if args.root_path is not None:
        root_path = args.root_path
    
        path = f"{root_path}/fakedata/epoch{epoch}/minibatch{minibatch}.h5"

        newpath = f"{root_path}/visualize/epoch{epoch}/"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
    else:
        path = None
    
    data_raw, data_raw_wpointwolf, data_fake, label= load_h5_new(h5_name=path)

    print(data_raw.shape[0])
    for i in range(data_raw.shape[0]):
        raw = data_raw[i]
        fake = data_fake[i]
        label_ = label[i]
        raw_wpointwolf = data_raw_wpointwolf[i]
        pointclouds = [
            {'name': 'raw', 'data': raw},
            {'name': 'fake', 'data': fake},
            {'name': 'raw_wpointwolf', 'data': raw_wpointwolf}
        ]
        # o3dvis(points_1=raw, points_2=raw_wpointwolf, points_3=fake, id=f"{i}_{classes[int(label_)]}")
        # o3dvis(rawpoints=raw, newpoints=fake, id=f"{i}_{classes[int(label_)]}")

    #     front_size = [-0.078881283843093356, 0.98265290049739873, -0.16784224796908237]
    # lookat_size = [0.057118464194894254, -0.010695673712330742, 0.047245129152854129]
    # up_size = [0.018854223129214469, -0.16686616421723113, -0.98579927039414161]

        camera_position = np.array([-0.5, -0.5, -0.5])
        look_at = np.array([0.057118464194894254, -0.010695673712330742, 0.047245129152854129])
        up_direction = np.array([0.018854223129214469, -0.16686616421723113, -0.98579927039414161])

        for pc in pointclouds:
            name = pc['name']
            data = pc['data']
            data = data[:, [0, 2, 1]]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plotting the point cloud
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='.', s=1)

            # Set camera position and orientation
            ax.view_init(elev=30, azim=45)  # Adjust elevation and azimuth angles as needed
            ax.set_xlim([-1, 1])  # Adjust limits as needed
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Save the rendered figure as an image (e.g., PNG format)
            plt.savefig(f'{newpath}/{i}_{classes[int(label_)]}_{name}.png', dpi=300)  # Adjust dpi as needed for image quality
            plt.close()


    # print('data_raw.shape:', data_raw.shape)
    # print('data_fake.shape:', data_fake.shape)
    # print('label.shape:', label.shape)








