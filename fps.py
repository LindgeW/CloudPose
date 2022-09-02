import torch
import numpy as np


# 保证对点云样本的均匀采样，思想：不断迭代地选择距离已有采样点集的最远点
def farthest_point_sample(xyz, npoint):
    '''
    xyz: B N C
    npoint: 最远点采样的样本数
    输出：
    最远点采样的结果: B npoint，即npoint个采样点在原始点云中的索引
    '''
    device = xyz.device
    B, N, C = xyz.shape
    # 初始化最远点采样点矩阵,可以多个batch一起进行最远点采样
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # 初始化采样点到所有点的距离矩阵，保存所有点到最远点采样矩阵的最大距离
    distance = torch.ones(B, N).to(device) * 1e10
    # 初始化batch_size数组
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # 计算输入点云的重心 B 1 C
    barycenter = torch.mean(xyz, dim=1, keepdim=True)
    # 计算所有点到重心的距离 B N C
    dist = torch.sum((xyz - barycenter) ** 2, -1)
    # 选择距离重心最远的点作为第一个点 B 1
    farthest = torch.argmax(dist, dim=1)

    for i in range(npoint):
        # centroid[:,i] B,1
        centroids[:, i] = farthest
        # 从xyz中取出最远点 B,1,C
        centr = xyz[batch_indices, farthest, :].reshape(B, 1, 3)
        # 计算所有点到最远点的距离 B, npoint, C
        dist = torch.sum((xyz - centr) ** 2, -1)
        # 然后更新所有点到最远点集合的距离矩阵，注意距离矩阵中维持的始终是所有点到最远点集合的最小距离
        mask = dist < distance
        distance[mask] = dist[mask]
        # 然后找出距离矩阵中的最大值对应的点对最远点集合进行更新
        farthest = torch.argmax(distance, dim=-1)
    return centroids


def farthest_point_sample_np(xyz, npoint=100):
    '''
    xyz: B N C
    npoint: 最远点采样的样本数
    输出：
    最远点采样的结果: B npoint，即npoint个采样点在原始点云中的索引
    '''
    B, N, C = xyz.shape
    # 初始化最远点采样点矩阵,可以多个batch一起进行最远点采样
    centroids = np.zeros((B, npoint), dtype=np.int32)
    # 初始化采样点到所有点的距离矩阵，保存所有点到最远点采样矩阵的最大距离
    distance = np.ones((B, N)) * 1e10
    # 初始化batch_size数组
    batch_indices = np.arange(B)

    # 计算输入点云的重心 B 1 C
    barycenter = xyz.mean(axis=1, keepdims=True)
    # 计算所有点到重心的距离 B N C
    dist = ((xyz - barycenter) ** 2).sum(axis=-1)
    # 选择距离重心最远的点作为第一个点 B 1
    farthest = dist.argmax(axis=1)

    for i in range(npoint):
        # centroid[:,i] B,1
        centroids[:, i] = farthest
        # 从xyz中取出最远点 B,1,C
        centr = xyz[batch_indices, farthest, :].reshape(B, 1, 3)
        # 计算所有点到最远点的距离 B, npoint, C
        dist = ((xyz - centr) ** 2).sum(axis=-1)
        # 然后更新所有点到最远点集合的距离矩阵，注意距离矩阵中维持的始终是所有点到最远点集合的最小距离
        mask = dist < distance
        distance[mask] = dist[mask]
        # 然后找出距离矩阵中的最大值对应的点对最远点集合进行更新
        farthest = distance.argmax(axis=-1)
    return centroids


if __name__ == '__main__':
    # 创建点云数据, 注意点云数据的维度对应为batch, N, channel
    sim_data = torch.rand(2, 8, 3)
    # 调用最远点采样函数，获得最远点
    centroid = farthest_point_sample(sim_data, 4)
    print(centroid)