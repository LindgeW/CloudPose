from torch.utils.data import Dataset
import numpy as np
import os
import sys
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'cloud_poser'))


def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None:
        replace = (pc.shape[0] < num_sample)
        # replace: True表示可以取相同数字，False表示不可以取相同数字
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]


MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5])  # sunrgbd color is in 0~1
class MyDataset(Dataset):
    def __init__(self, split_set='train', num_points=1000,
                 use_color=False):
        assert(num_points <= 4096)
        self.data_path = os.path.join(BASE_DIR, 'scans/my_%s' % split_set)
        self.scans = sorted(list(set([os.path.basename(x) for x in os.listdir(self.data_path) if 'scan' in x])))
        self.poses = sorted(list(set([os.path.basename(x) for x in os.listdir(self.data_path) if 'pose' in x])))
        assert len(self.scans) == len(self.poses)
        self.num_points = num_points
        self.use_color = use_color

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        scan_name = self.scans[idx]
        pose_name = self.poses[idx]
        point_cloud = np.loadtxt(os.path.join(self.data_path, scan_name))
        pose = np.loadtxt(os.path.join(self.data_path, pose_name))

        if self.use_color is False:
            point_cloud = point_cloud[:, 0:3]
        else:
            point_cloud = point_cloud[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB)

        print(len(point_cloud))
        # ------------------------------- LABELS ------------------------------
        ret_dict = {}
        point_cloud = random_sampling(point_cloud, self.num_points)
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        matrix33 = pose[:3, :3]
        axag = cv2.Rodrigues(matrix33)[0].flatten()   # 旋转矩阵转旋转向量
        translate = pose[:3, 3]
        ret_dict['axag_label'] = axag.astype(np.float32)
        ret_dict['translate_label'] = translate.astype(np.float32)
        return ret_dict