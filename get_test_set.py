import os.path
import random
from fps import farthest_point_sample_np
import numpy as np
import open3d as o3d
from glob import glob
from conf import K

# 得到测试集（包括点云和位姿RT）

width, height, fx, fy, cx, cy = K['width'], K['height'], K['fx'], K['fy'], K['cx'], K['cy']


def sample_data(dir, num_test=1, train_path=None):
    rgb_imgs, depth_imgs, masks, boxes, poses = [], [], [], [], []
    if train_path is None:
        rgb_paths = random.sample(glob(os.path.join(dir, '*-color.png')), num_test)
        # rgb_paths = glob(os.path.join(dir, '*-color.png'))[:num_test]
    else:
        rgb_paths = []
        train_paths = []
        with open(train_path, 'r') as fin:
            for pt in fin:
                train_paths.append(pt.strip())
        for path in glob(os.path.join(dir, r'*-color.png')):
            if path not in train_paths:
                rgb_paths.append(path)
                if len(rgb_paths) == num_test:
                    break

    depth_paths = [rp.replace('-color.png', '-depth.png') for rp in rgb_paths]
    mask_paths = [rp.replace('-color.png', '-label.png') for rp in rgb_paths]
    box_paths = [rp.replace('-color.png', '-box.txt') for rp in rgb_paths]
    pose_paths = [rp.replace('-color.png', '-pose.txt') for rp in rgb_paths]
    for rgb_path, depth_path, mask_path, box_path, pose_path in zip(rgb_paths, depth_paths, mask_paths, box_paths, pose_paths):
        rgb_map = o3d.io.read_image(rgb_path)
        depth_map = o3d.io.read_image(depth_path)
        mask_map = o3d.io.read_image(mask_path)
        rgb_imgs.append(rgb_map)
        depth_imgs.append(depth_map)
        masks.append(np.asarray(mask_map, dtype=bool))
        boxes.append(np.loadtxt(box_path, delimiter='\n', dtype=int))
        poses.append(np.loadtxt(pose_path))
    return rgb_imgs, depth_imgs, masks, boxes, poses


def get_pc_bb(depth_map, box):
    top, left, w, h = box
    # 完整点云
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_map, intrinsic, project_valid_depth_only=False)
    full_pc = np.asarray(pcd.points).reshape((height, width, 3))
    # Object点云
    obj_pc = o3d.geometry.PointCloud()
    obj_pc.points = o3d.utility.Vector3dVector(full_pc[left: left+h, top: top+w].reshape(-1, 3))
    obj_pc = obj_pc.remove_statistical_outlier(nb_neighbors=50, std_ratio=2)[0]
    return obj_pc


def get_pc_seg(depth_map, seg):
    # 完整点云
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_map, intrinsic, project_valid_depth_only=False)
    full_pc = np.asarray(pcd.points).reshape((height, width, 3))
    # Object点云
    obj_pc = o3d.geometry.PointCloud()
    obj_pc.points = o3d.utility.Vector3dVector(full_pc[seg].reshape(-1, 3))
    obj_pc = obj_pc.remove_statistical_outlier(nb_neighbors=50, std_ratio=2)[0]
    return obj_pc


def run():
    # rgbs, depths, masks, boxes, poses = sample_data('../dataset/ape', 10)
    rgbs, depths, masks, boxes, poses = sample_data('../dataset/ape', 10, 'train_imgs.txt')
    bb = False
    for i in range(len(rgbs)):
        if bb:
            obj_pc = get_pc_bb(depths[i], boxes[i])
        else:
            obj_pc = get_pc_seg(depths[i], masks[i])

        o3d.io.write_point_cloud(f'scans/my_val/scan_{i}.xyz', obj_pc)
        np.savetxt(f'scans/my_val/pose_{i}.txt', poses[i])

    print('Test Set Done !')

run()