import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import STN3d, STNkd, feature_transform_reguliarzer


class get_model(nn.Module):
    # part_num：表示要分割的部分数量
    # normal_channel：表示是否使用法线通道。如果为True，则通道数为6（XYZ + 法线），否则为3（仅XYZ）。
    def __init__(self, part_num=50, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        # 设置类的part_num属性
        # 初始化一个3D空间变换网络（STN3d）
        self.part_num = part_num
        self.stn = STN3d(channel)
        # 初始化一系列的1D卷积层和批标准化层
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        # 初始化一个k维空间变换网络（STNkd），其中k=128
        self.fstn = STNkd(k=128)
        # 初始化另一系列的1D卷积层和批标准化层
        self.convs1 = torch.nn.Conv1d(4944, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, part_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    # 定义前向传播函数，接受点云数据和标签作为输入
    def forward(self, point_cloud, label):
        # 获取点云数据的大小（B：批次大小，D：特征维度，N：点数量）
        B, D, N = point_cloud.size()
        # 使用STN对点云进行空间变换。
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        # 如果特征维度D大于3，将点云数据分割成坐标部分和额外特征部分
        # 使用split函数沿着第三个维度（dim=2，索引从0开始）将point_cloud分割成两部分。
        # 第一部分是三维坐标，存储在point_cloud中，而其余的特征存储在feature中。
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        # 使用torch.bmm（batch matrix multiplication，批矩阵乘法）对三维坐标应用空间变换trans。
        # 这里的trans通常是由空间变换网络（STN）学习得到的，用于对点云数据进行旋转、平移等空间变换，以便更好地对齐和标准化数据。
        point_cloud = torch.bmm(point_cloud, trans)
        # 再次检查D是否大于3。如果是，我们需要将之前分割出来的额外特征feature重新与变换后的三维坐标point_cloud组合在一起
        # 使用torch.cat函数沿着第三个维度（dim=2）将point_cloud和feature拼接起来。这样，我们就得到了一个包含变换后的三维坐标和原始额外特征的新点云张量
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)

        # 再次转置点云数据的维度
        point_cloud = point_cloud.transpose(2, 1)

        # 通过一系列的卷积层和激活函数进行特征提取
        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        trans_feat = self.fstn(out3)
        # 将 out3 的第二和第三个维度交换，为了满足后续矩阵乘法的需求
        x = out3.transpose(2, 1)
        # 执行批量矩阵乘法，将 x 和 trans_feat 相乘。
        net_transformed = torch.bmm(x, trans_feat)
        # 相乘完后，再次转置，将结果调整回期望的形状。
        net_transformed = net_transformed.transpose(2, 1)

        # 对 `net_transformed` 进行进一步的卷积和批标准化操作，然后应用 ReLU 激活函数
        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        # 执行最大池化操作，沿着第三个维度（索引为2）提取最大值，同时保持维度不变（`keepdim=True`）
        out_max = torch.max(out5, 2, keepdim=True)[0]
        # 让张量拉直
        out_max = out_max.view(-1, 2048)

        # 这行代码将 out_max 和 label 沿着第二个维度（索引为1的维度）拼接在一起
        # label.squeeze(1) 用于移除 label 中大小为1的维度
        # 如果 label 的形状是 [B, 1, ...]，那么 squeeze(1) 之后的形状就会变成 [B, ...]。拼接后的结果存储在 out_max 中
        out_max = torch.cat([out_max,label.squeeze(1)],1)
        # 将 out_max 重新塑形，其中 -1 表示该维度的大小由其他维度的大小和元素总数自动计算，2048+16 可能是因为前面将 out_max 和一个有16个特征的 label 拼接在了一起
        expand = out_max.view( -1, 2048+16, 1).repeat(1, 1, N)
        # 这行代码将 expand、out1、out2、out3、out4 和 out5 沿着第二个维度拼接在一起，拼接后的结果存储在 concat 中
        # 分割任务需要把全局特征和局部特征拼接在一起
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        # 这三行代码是卷积层和批标准化层的堆叠，每行都先应用卷积然后应用批标准化，最后应用 ReLU 激活函数
        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        # 应用最后一层卷积，不应用激活函数和批标准化
        net = self.convs4(net)
        # 将第二和第三个维度交换。
        # contiguous()：确保张量在内存中是连续存储的，这在某些张量操作之后是必要的。
        net = net.transpose(2, 1).contiguous()
        # net.view(-1, self.part_num)：将 net 重新塑形为两个维度，其中第二个维度的大小是 self.part_num，第一个维度的大小由元素总数和第二个维度的大小自动计算。
        # F.log_softmax(...)：将重新塑造后的 net 在最后一个维度上应用对数 softmax(dim=-1 指定了要在哪个维度上计算 softmax)，得到每个类别的对数概率。
        net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        # 这行代码再次使用 view() 方法来重新塑造 net 的形状
        # 经过上一行代码的处理后，net 的形状是 [D, self.part_num]。现在，它被重新塑造为 [B, N, self.part_num]
        # B 通常是批次大小（batch size），表示一次处理的样本数
        # N 可能是每个样本中的元素数量或特征数量，具体取决于上下文
        # self.part_num 是类别的数量，这里假设为 50
        net = net.view(B, N, self.part_num) # [B, N, 50]

        return net, trans_feat


class get_loss(torch.nn.Module):
    # mat_diff_loss_scale: 这是一个可选参数，用于缩放mat_diff_loss（矩阵差异损失）
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    # 这个方法接受三个参数：pred（模型的预测输出），target（真实标签），和trans_feat（转换特征，用于计算mat_diff_loss）
    def forward(self, pred, target, trans_feat):
        # 使用负对数似然损失（Negative Log Likelihood Loss）计算pred和target之间的损失。这通常用于分类问题
        loss = F.nll_loss(pred, target)
        # 计算trans_feat的正则化损失，为了确保特征转换不会过于极端或不稳定
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        # 它是nll_loss和mat_diff_loss的加权和
        # mat_diff_loss被mat_diff_loss_scale缩放
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss