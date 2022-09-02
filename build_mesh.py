import os.path
import random
from fps import farthest_point_sample_np
import numpy as np
import open3d as o3d
from glob import glob
from conf import K

width, height, fx, fy, cx, cy = K['width'], K['height'], K['fx'], K['fy'], K['cx'], K['cy']


def save_train_rgbs(path, rgbs):
    assert isinstance(rgbs[0], str)
    with open(path, 'w') as fw:
        for rgb in rgbs:
            fw.write(rgb)
            fw.write('\n')
    print('Save RGB Done!')


def sample_data(dir, n=10):
    rgb_imgs, depth_imgs, masks, boxes, poses = [], [], [], [], []
    rgb_paths = random.sample(glob(os.path.join(dir, '*-color.png')), n)
    save_train_rgbs('train_imgs.txt', rgb_paths)
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
    rgbs, depths, masks, boxes, poses = sample_data('../dataset/ape', 10)
    bb = False
    # merge_pc = []
    n = 1500  # 每个点云采样N个点
    mesh = np.zeros((len(rgbs) * n, 3), dtype=np.float32)
    fps_mesh = np.zeros((len(rgbs) * n, 3), dtype=np.float32)
    for i in range(len(rgbs)):
        if bb:
            obj_pc = get_pc_bb(depths[i], boxes[i])
        else:
            obj_pc = get_pc_seg(depths[i], masks[i])

        # 相机坐标系下的点云统一到物体坐标系
        # method1: SE
        # pose = np.append(poses[i], [[0, 0, 0, 1]], axis=0)
        # obj_pc.transform(np.linalg.inv(pose))
        # merge_pc.append(obj_pc.points)

        # method2: minus t and times R^-1
        points = (obj_pc.points - poses[i][:, 3]) @ poses[i][:, :3]
        assert len(points) > 0
        sample_ids = np.random.choice(len(points), n, replace=True)
        mesh[i*n: (i+1)*n, :] = points[sample_ids, :]
        fps_sample_ids = farthest_point_sample_np(points.reshape(1, -1, 3), n).reshape(-1)
        fps_mesh[i * n: (i + 1) * n, :] = points[fps_sample_ids, :]

    merge_pcd = o3d.geometry.PointCloud()
    # merge_pcd.points = o3d.utility.Vector3dVector(np.concatenate(merge_pc, axis=0))
    merge_pcd.points = o3d.utility.Vector3dVector(mesh)
    merge_pcd = merge_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2)[0]
    o3d.visualization.draw_geometries([merge_pcd],
                                      window_name="Object Point Cloud",
                                      width=1000,
                                      height=800)

    fps_merge_pcd = o3d.geometry.PointCloud()
    fps_merge_pcd.points = o3d.utility.Vector3dVector(fps_mesh)
    fps_merge_pcd = fps_merge_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2)[0]
    o3d.io.write_point_cloud('depth2pc.xyz', fps_merge_pcd)

    fps_merge_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=20))
    o3d.visualization.draw_geometries([fps_merge_pcd],
                                      window_name="Object Point Cloud",
                                      width=1000,
                                      height=800,
                                      point_show_normal=True)

run()