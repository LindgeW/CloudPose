import numpy as np
from scipy import spatial, linalg


# 5cm, 5°
def cm_degree_5_metric(pose_pred, pose_target):
    translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_target[:, 3]) * 100  # m -> cm
    rotation_diff = pose_pred[:, :3] @ pose_target[:, :3].T
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
    print(translation_distance, angular_distance)
    return translation_distance < 5 and angular_distance < 5


def calc_rt_dist_m(pose_src, pose_tgt):
    R_src = pose_src[:, :3]
    T_src = pose_src[:, 3]
    R_tgt = pose_tgt[:, :3]
    T_tgt = pose_tgt[:, 3]
    temp = linalg.logm(np.dot(np.transpose(R_src), R_tgt))
    rd_rad = np.linalg.norm(temp, 'fro') / np.sqrt(2)
    rd_deg = 180 * rd_rad / np.pi
    td = np.linalg.norm(T_tgt - T_src) * 100   # m -> cm
    return rd_deg < 5 and td < 5


def trans_err_eval(poses_pred, poses_tgt):
    cmd = []
    for pp, pt in zip(poses_pred, poses_tgt):
        if pp.shape == (4, 4):
            pp = pp[:3, :4]
        if pt.shape == (4, 4):
            pt = pt[:3, :4]
        cmd.append(cm_degree_5_metric(pp, pt))
    return cmd, 100.*np.mean(cmd)


def transform_pts_Rt(pts, R, t):
    """
    Applies a rigid transformation to 3D points.
    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert (pts.shape[1] == 3)
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T


def add(R_est, t_est, R_gt, t_gt, pts):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e


def adi(R_est, t_est, R_gt, t_gt, pts):
    """
    Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)
    # Calculate distances to the nearest neighbors from pts_gt to pts_est
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)
    e = nn_dists.mean()
    return e


def pose_add(poses_pred, poses_gt, model_pts, cls_name='ape', obj_diameter=102.099):  # mm
    count_correct = {k: 0 for k in ['0.02', '0.05', '0.10']}
    threshold_002 = 0.02 * obj_diameter / 1000
    threshold_005 = 0.05 * obj_diameter / 1000
    threshold_010 = 0.1 * obj_diameter / 1000
    count_all = len(poses_gt)
    for j in range(count_all):
        RT = poses_pred[j]     # est pose
        pose_gt = poses_gt[j]  # gt pose
        if cls_name == 'eggbox' or cls_name == 'glue' or cls_name == 'bowl' or cls_name == 'cup':
            error = adi(RT[:3, :3], RT[:, 3], pose_gt[:3, :3], pose_gt[:, 3], model_pts)
        else:
            error = add(RT[:3, :3], RT[:, 3], pose_gt[:3, :3], pose_gt[:, 3], model_pts)

        if error < threshold_002:
            count_correct['0.02'] += 1
        if error < threshold_005:
            count_correct['0.05'] += 1
        if error < threshold_010:
            count_correct['0.10'] += 1

    acc = {'0.02': 100 * count_correct['0.02'] / count_all,
           '0.05': 100 * count_correct['0.05'] / count_all,
           '0.10': 100 * count_correct['0.10'] / count_all}
    return acc
