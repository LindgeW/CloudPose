import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
# https://github.com/wawatt/CloudPose_pt/


class CloudPose_trans(nn.Module):
    def __init__(self, channel=3):
        super(CloudPose_trans, self).__init__()
        self.in_feature_dim = channel
        self.conv1 = torch.nn.Conv1d(self.in_feature_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

    def forward(self, point_cloud):
        batch_size = point_cloud.shape[0]
        # num_point = point_cloud.shape[2]
        x = F.relu(self.bn1(self.conv1(point_cloud)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        max_indices = torch.argmax(x, dim=1)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.reshape(batch_size, 1024)
        x = F.relu(self.bn6(self.fc1(x)))
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.fc3(x)
        return x, max_indices


class CloudPose_rot(nn.Module):
    def __init__(self, channel=3):
        super(CloudPose_rot, self).__init__()
        self.in_feature_dim = channel
        self.conv1 = torch.nn.Conv1d(self.in_feature_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

    def forward(self, point_cloud):
        batch_size = point_cloud.shape[0]
        # num_point = point_cloud.shape[2]
        x = F.relu(self.bn1(self.conv1(point_cloud)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        max_indices = torch.argmax(x, dim=1)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.reshape(batch_size, 1024)
        x = F.relu(self.bn6(self.fc1(x)))
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.fc3(x)
        return x, max_indices


class CloudPose_all(nn.Module):
    def __init__(self, channel=3):
        super(CloudPose_all, self).__init__()
        self.channel = channel
        self.trans = CloudPose_trans(self.channel)
        self.rot = CloudPose_rot(self.channel)

    def forward(self, inp):
        point_clouds = inp['point_clouds']
        point_clouds_tp = point_clouds.transpose(1, 2)  # b 3 256
        base_xyz = torch.mean(point_clouds_tp[:, :self.channel, :], dim=2)
        point_clouds_res = point_clouds_tp[:, :self.channel, :] - base_xyz.unsqueeze(-1)  # b 3 1
        t, ind_t = self.trans(point_clouds_res)
        r, ind_r = self.rot(point_clouds_res)   # better than point_clouds_tp
        end_points = {'translate_pred': t + base_xyz, 'axag_pred': r}
        return end_points
