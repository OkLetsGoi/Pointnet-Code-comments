import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# 该类是3D空间变换网络（Spatial Transformer Network, STN）的实现
# STN是一个可以插入到现有神经网络架构中的模块，用于明确地学习数据的空间变换，从而改善模型的性能
# 这里的3D版本专门用于处理三维数据
# 这行代码定义了一个继承自nn.Module的新类STN3d
# nn.Module是PyTorch中所有神经网络模块的基类
class STN3d(nn.Module):
    # 初始化函数__init__接收一个参数channel，该参数指定输入数据的通道数
    # 调用父类nn.Module的初始化函数
    def __init__(self, channel):
        super(STN3d, self).__init__()
        # 这里定义了三个一维卷积层，用于从输入数据中提取特征
        # 每个卷积层的输出通道数分别为64、128和1024，而卷积核的大小都是1
        # 1x1卷积核ke'y改变输出特征图通道数(视频链接:https://www.bilibili.com/video/BV11m4y1r7jK?vd_source=716661182e74024d94c94830e65afe1b)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # 定义了三个全连接层（或线性层）。这些层用于进一步处理卷积层提取的特征，并最终输出一个9维的向量，该向量表示3D仿射变换矩阵的参数
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        # 最后得到的9个特征,会变成3x3矩阵,用以矩阵乘法
        self.fc3 = nn.Linear(256, 9)
        # 定义一个ReLU激活函数
        self.relu = nn.ReLU()

        # 定义五个批标准化层，用于加速训练过程并提高模型的稳定性
        # 每个实例都接受一个整数参数，该参数指定了输入特征的维度（即通道数）
        # 如self.bn1 = nn.BatchNorm1d(64)：这行代码创建了一个批量归一化层，用于处理具有 64 个通道的一维输入数据
        # 参考链接:https://www.bilibili.com/video/BV1Lx411j7GT?vd_source=716661182e74024d94c94830e65afe1b
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    # 定义前向传播函数forward，它接收一个输入张量x并返回变换后的张量
    def forward(self, x):
        # 获取输入数据的批量大小
        batchsize = x.size()[0]
        # 应用卷积层、批标准化层和ReLU激活函数
        # 经过三个一维卷积,输出维度变为1024
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # 对输入张量 x 沿着第三个维度（索引为2，因为维度索引从0开始）进行最大值操作，并且保留最大值的维度（即不减少维度的数量）
        # keepdim=True：这个参数指定是否保留原始张量的维度
        # 如果设置为 True，则输出张量的维度将与输入张量相同，除了进行最大值操作的那个维度，它将变为1。如果设置为 False，则输出张量的维度将减少一个
        # 网络中的最大池化,即可得到全局特征
        x = torch.max(x, 2, keepdim=True)[0]
        # x.view(): 这是改变张量形状的方法
        # -1: 这是一个特殊的值，意味着该维度的大小将自动计算，以便总元素数量保持不变
        # 1024: 我们希望第二个维度的大小为1024
        # view不会更改张量的数据，只是更改“视图”,这里相当于把张量拉直(1行1024列)
        x = x.view(-1, 1024)

        # 应用全连接层、批标准化层和ReLU激活函数（最后一层除外，因为它输出的是仿射变换参数，不需要激活函数）
        # 经过三个全连接,输出9个元素
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)：这里首先使用 NumPy 创建一个 1x9 的数组，表示一个 3x3 的恒等矩阵（按行展开）。
        # 然后使用 astype(np.float32) 将数组的数据类型转换为 32 位浮点数，这是因为 PyTorch 通常使用浮点数进行计算。
        # torch.from_numpy(...)：这个函数将 NumPy 数组转换为 PyTorch 张量（tensor），这样它就可以在 PyTorch 框架中使用了。
        # view(1, 9)：这个方法将张量的形状改变为 (1, 9)，确保它是一个二维张量，其中第一维是 1（表示单个矩阵），第二维是 9（表示矩阵的元素数量）。
        # repeat(batchsize, 1)：这个方法将张量沿着第一个维度重复 batchsize 次，而沿着第二个维度保持不变。
        # 这是为了匹配输入数据的批量大小，确保每个输入样本都有一个对应的恒等变换矩阵。
        # Variable(...)：在早期的 PyTorch 版本中，Variable 是用来包装张量并允许对其进行自动求导的类
        # 但从 PyTorch 0.4.0 开始，Variable 类与 torch.Tensor 类合并,因此直接使用张量就可以享受自动求导功能，无需显式地将张量包装为 Variable
        # 最终，iden 是一个形状为 (batchsize, 9) 的张量，其中每个元素都是一个 3x3 恒等变换矩阵的展开形式
        # 这个张量将用作后续操作的基准，与网络学习到的仿射变换参数相加，以确保在没有学习到任何有用的变换时，网络至少能够保持输入不变
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        # 如果输入数据在GPU上，则将恒等变换矩阵也移动到GPU上
        if x.is_cuda:
            iden = iden.cuda()
        # 将学习到的变换参数加到恒等变换矩阵上，并将结果张量调整为形状(batch_size, 3, 3)，这样每个元素都是一个3x3的仿射变换矩阵
        # 实现仿射变换
        # 仿射变换在神经网络中的主要目的是实现线性空间中的旋转、平移和缩放变换，从而增加模型的灵活性和表达能力
        # 通过仿射变换，神经网络可以学习到输入数据的不同几何变化特征，使得训练出来的模型能够适应不同的变化场景
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


# 前面那个函数是3维度,这个函数时k维
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # 保存传入参数k的值
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    #
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        # 初始化一个STN3d对象，用于空间变换网络
        self.stn = STN3d(channel)
        # 初始化三个一维卷积层
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # 初始化三个批量归一化层
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        # 保存传入的参数值
        # global_feat 表示是否需要全局特征
        # feature_transform 表示是否需要特征转换
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        # 如果feature_transform为True，则初始化另一个空间变换网络STNkd
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        # 获取输入x的尺寸，其中B是batch_size，D是特征维度（3（xyz坐标）或6（xyz坐标+法向量）），N是一个物体所取点的数量
        B, D, N = x.size()
        # 通过STN3d获取空间变换矩阵
        trans = self.stn(x)
        # 交换x的第二和第三维度
        x = x.transpose(2, 1)
        # 如果特征维度大于3，将额外的特征与前三维特征分开
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        # x与变换矩阵相乘，对x应用空间变换
        # torch.bmm 是 PyTorch 中的一个函数，用于执行批量矩阵乘法（batch matrix multiplication）
        # 具体来说，torch.bmm(x, y) 对 x 和 y 中的每一对矩阵执行矩阵乘法，其中 x 和 y 都是三维张量（tensors），并且它们的第三维（从0开始计数）必须匹配，以便进行矩阵乘法。
        # x 是一个三维张量，我们可以假设它的维度是 (B, N, D)，其中 B 是批量大小，N 是点的数量，D 是每个点的特征维度（在前面的代码中可能已经被变换过，例如通过空间变换网络）。
        # trans 也是一个三维张量，但它可能具有不同的维度，比如 (B, D, D) 或 (B, D, K)，具体取决于它是如何计算得出的。
        # torch.bmm(x, trans) 执行批量矩阵乘法，将 x 中的每个 (N, D) 矩阵与 trans 中的对应 (D, D) 矩阵相乘，
        # 结果是一个新的三维张量，其维度为 (B, N, D)（假设 trans 的维度是 (B, D, D)）。
        # B为bath_size
        x = torch.bmm(x, trans)
        # 如果特征维度D大于3（通常3代表三维空间中的x, y, z坐标），则将额外的特征feature拼接到x上
        # 拼接是在第三个维度（dim=2）上进行的
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        # 如果启用了特征变换（feature_transform为True），则使用fstn来计算特征变换矩阵trans_feat
        # 然后，对x进行转置，执行批量矩阵乘法（torch.bmm）以应用这个特征变换，再转置回来。
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        # 将当前的x保存为pointfeat（可能是为了后续使用或返回）,pointfeat为局部特征
        # 对x应用另外两个卷积层和批量归一化层，第二个卷积层后还有ReLU激活
        pointfeat = x
        # 获得局部特征，再做MLP
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # 使用最大池化来提取全局特征
        # 这里假设x的第三个维度（dim=2）是空间维度或点云中的点数，通过取最大值来聚合所有点的信息
        x = torch.max(x, 2, keepdim=True)[0]
        # 将全局特征展平
        x = x.view(-1, 1024)
        # 看是否需要返回全局特征（即是否启用全局特征提取，global_feat为True），是则返回全局特征x、空间变换矩阵trans和特征变换矩阵trans_feat
        if self.global_feat:
            return x, trans, trans_feat
        # 否，拼接任务中，需要返回局部特征与全局特征的拼接，则将全局特征x与局部特征pointfeat拼接，然后返回这个拼接后的特征以及trans和trans_feat。
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

# 对特征转换矩阵进行正则化(希望feature_transform矩阵接近于正交阵,才不会损失特征信息)
# 定义损失函数让它训练的过程中接近于正交矩阵
def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    # torch.eye(d) 创建一个 D x D 的单位矩阵（对角线上为1，其余为0）
    # [None, :, :]将这个单位矩阵增加一个额外的维度，变成 1 x D x D，以便与trans进行批量矩阵乘法。
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    # trans.transpose(2, 1)：将 `trans` 的后两个维度转置，使得批量矩阵乘法的维度匹配。
    # torch.bmm(trans, trans.transpose(2, 1))：执行批量矩阵乘法。这计算了 `trans` 和它的转置之间的乘积。
    # - I：从上述乘积中减去单位矩阵。如果 `trans` 是正交矩阵，那么这个差应该是零矩阵。
    # torch.norm(..., dim=(1, 2))：计算每个矩阵的Frobenius范数（即所有元素的平方和的平方根），这衡量了矩阵与零矩阵的距离。
    # torch.mean(...)：最后，计算所有矩阵范数的平均值，得到正则化损失。
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    # 返回正则化损失函数
    return loss
