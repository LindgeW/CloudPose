import open3d as o3d
from scipy.spatial.transform import Rotation as Rot
import numpy as np
from conf import K

width, height, fx, fy, cx, cy = K['width'], K['height'], K['fx'], K['fy'], K['cx'], K['cy']


def filter_pc_bg(pc, eps=1e-10):   # 注：传入的是相机坐标系下的点云
    # 滤除镜头后方的点
    valid = pc[:, 2] > eps
    z = pc[valid, 2]
    # (相机坐标系下)点云反向映射到像素坐标位置
    u = np.round(pc[valid, 0] * fx / z + cx).astype(int)
    v = np.round(pc[valid, 1] * fy / z + cy).astype(int)
    # 滤除超出图像尺寸的无效像素
    valid = np.bitwise_and(np.bitwise_and((u >= 0), (u < width)),
                           np.bitwise_and((v >= 0), (v < height)))
    u, v, z = u[valid], v[valid], z[valid]
    # 按距离填充生成深度图，近距离覆盖远距离
    img_z = np.full((height, width), np.inf)
    for ui, vi, zi in zip(u, v, z):
        img_z[vi, ui] = min(img_z[vi, ui], zi)    # 近距离像素屏蔽远距离像素
    # 小洞和“透射”消除
    img_z_shift = np.array([img_z,
                            np.roll(img_z, 1, axis=0),
                            np.roll(img_z, -1, axis=0),
                            np.roll(img_z, 1, axis=1),
                            np.roll(img_z, -1, axis=1)])
    img_z = np.min(img_z_shift, axis=0)
    front_pc = []
    for ui, vi in zip(u, v):
        z = img_z[vi, ui]
        x = z * (ui - cx) / fx
        y = z * (vi - cy) / fy
        front_pc.append((x, y, z))
    print(front_pc)
    return np.asarray(front_pc)


# 如何避免从物体斜下方去拍？
def get_trans(N=1, seed=1347):
    np.random.seed(seed)
    R = Rot.random(N).as_matrix()  # N x 3 x 3
    T = np.random.randn(N, 3)  # N x 3
    T[:, 0] = T[:, 0].clip(min=-1, max=1)  # N x 3
    T[:, 1] = T[:, 1].clip(min=-1, max=1)  # N x 3
    T[:, 2] = T[:, 2].clip(min=0.7, max=1)  # N x 3
    return R, T


def pc_scan(mesh, num=1):
    if isinstance(mesh, str):
        pcd = o3d.io.read_point_cloud(mesh)
    else:
        pcd = mesh

    o3d.visualization.draw_geometries([pcd],
                                      window_name="Object Cloud",
                                      width=1000,
                                      height=800,
                                      left=100, top=100,
                                      point_show_normal=True)

    M = get_trans(num)
    for i, (R, T) in enumerate(zip(*M)):
        # fg_pcd = o3d.geometry.PointCloud()
        # fg_pcd.points = o3d.utility.Vector3dVector(filter_pc_bg((R @ np.array(pcd.points).T).T + T))
        # # fg_pcd = fg_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2)[0]
        # o3d.visualization.draw_geometries([fg_pcd],
        #                                   window_name="Object Cloud",
        #                                   width=1000,
        #                                   height=800,
        #                                   left=100, top=100,
        #                                   point_show_normal=True)

        cam_pcd = o3d.geometry.PointCloud()
        cam_pcd.points = o3d.utility.Vector3dVector((R @ np.asarray(pcd.points).T).T)
        D = np.linalg.norm(np.asarray(cam_pcd.get_max_bound()) - np.asarray(cam_pcd.get_min_bound()))
        diameter = np.linalg.norm(T)
        if diameter < D:
            continue
        # 设置隐点移除参数
        camera = [0, 0, -diameter]  # 视点位置
        radius = diameter * 100     # The radius of the spherical projection (球面投影的半径)
        pt_map = cam_pcd.hidden_point_removal(camera, radius=radius)[1]
        pc_visible = cam_pcd.select_by_index(pt_map)
        o3d.io.write_point_cloud(f'scans/my_train/scan_{i}.xyz', pc_visible)
        np.savetxt(f'scans/my_train/pose_{i}.txt', np.concatenate((R, T.reshape(-1, 1)), axis=1))
        # o3d.visualization.draw_geometries([pc_visible],
        #                                   window_name="Single-View Object Cloud",
        #                                   width=1000,
        #                                   height=800,
        #                                   left=100,
        #                                   top=100,
        #                                   point_show_normal=True)


pc_scan('depth2pc.xyz', 1000)
