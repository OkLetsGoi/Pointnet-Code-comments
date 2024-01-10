"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm
from data_utils.ShapeNetDataLoader import PartNormalDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_part_seg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')

    return parser.parse_args()


def main(args):
    # 定义一个log_string函数，它接受一个字符串参数str，并使用预先定义的logger（稍后在代码中定义）将信息记录到日志中，同时在控制台打印相同的字符串
    def log_string(str):
        logger.info(str)
        print(str)

    # 设置环境变量CUDA_VISIBLE_DEVICES，以指定哪些GPU设备可用于此脚本。args.gpu是从命令行参数中获取的GPU设备ID
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    # 使用pathlib模块中的Path类来创建一个代表'./log/'路径的Path对象，并将其赋值给变量exp_dir
    # './log/'表示当前目录下的log子目录
    exp_dir = Path('./log/')
    # exist_ok=True表示如果目录已经存在，则不会引发错误。如果设置为False（默认值）且目录已存在，则会引发FileExistsError
    exp_dir.mkdir(exist_ok=True)
    # 使用joinpath方法将part_seg这个子路径添加到exp_dir所代表的路径之后，并返回一个新的Path对象
    # 这样，exp_dir现在代表'./log/part_seg'这个路径
    exp_dir = exp_dir.joinpath('part_seg')

    exp_dir.mkdir(exist_ok=True)
    # 如果args.log_dir为None，则将当前时间字符串添加到exp_dir路径中，以创建一个唯一的子目录
    # 否则，使用args.log_dir作为子目录的名称。然后创建该子目录
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    # 在exp_dir下创建checkpoints和logs子目录，用于存储模型检查点和日志文件
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    # 调用parse_args()函数（该函数应在此代码段之前定义）来解析命令行参数，并将结果存储在args变量中
    args = parse_args()
    # 获取一个名为"Model"的logger对象。如果该名称的logger对象已经存在，则直接返回该对象；
    # 如果不存在，则创建一个新的logger对象
    logger = logging.getLogger("Model")
    # 设置logger对象的日志级别为INFO
    # 这意味着，所有级别为INFO及以上的日志消息（如INFO、WARNING、ERROR、CRITICAL）都将被处理，而级别低于INFO的（如DEBUG）则会被忽略
    logger.setLevel(logging.INFO)
    # 创建一个日志格式化器formatter，它定义了日志消息的格式。这里的格式字符串表示每条日志消息将包含以下信息：
    # %(asctime)s：日志消息产生的时间
    # %(name)s：logger对象的名称，这里是"Model"
    # %(levelname)s：日志消息的级别（如INFO、WARNING等）
    # %(message)s：实际的日志消息内容
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 创建一个文件处理器file_handler，用于将日志消息写入到文件中。
    # 文件的路径由log_dir和args.model两个变量拼接而成，例如，如果log_dir是'./logs'，args.model是'MyModel'，则日志将被写入到./logs/MyModel.txt文件中。
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    # 设置文件处理器的日志级别为INFO。这意味着，所有级别为INFO及以上的日志消息都将被写入到文件中。
    file_handler.setLevel(logging.INFO)
    # 将之前创建的格式化器formatter设置给文件处理器file_handler，这样file_handler就知道如何格式化日志消息了
    file_handler.setFormatter(formatter)
    # 将文件处理器file_handler添加到logger对象中。
    # 这样，当调用logger对象的日志记录方法（如logger.info()）时，相关的日志消息就会被传递给file_handler，并最终被写入到指定的文件中。
    logger.addHandler(file_handler)
    # 使用log_string函数记录参数信息
    log_string('PARAMETER ...')
    log_string(args)

    # 设置数据集根目录的路径
    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    # 创建训练数据集
    # `root`: 数据集的根目录。
    # `npoints`: 每个数据点云中需要采样的点数。
    # `split='trainval'`: 表示这个数据集是用于训练和验证的。
    # `normal_channel`: 指示是否使用法线通道。
    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    # 使用PyTorch的DataLoader来加载训练数据
    # `batch_size`: 每个批次的数据量大小。
    # `shuffle=True`: 在每个训练周期开始时，随机打乱数据。
    # `num_workers=10`: 使用10个子进程来加载数据，可以加速数据加载。
    # `drop_last=True`: 如果数据集的大小不能被`batch_size`整除，则丢弃最后一个批次的数据。
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    # 创建测试数据集
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    # 使用PyTorch的DataLoader来加载测试数据
    # shuffle = False，因为测试时不需要打乱数据
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    # 设置类别数和部件数
    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''
    # 使用importlib.import_module方法动态地导入一个模块。模块的名称由args.model指定
    # 例如，如果args.model的值为"my_model"，那么这行代码的效果等同于import my_model as MODEL
    MODEL = importlib.import_module(args.model)
    # 从models目录下复制一个名为args.model.py的文件到exp_dir所代表的目录中
    # 例如，如果args.model的值为"my_model"，那么它会复制models/my_model.py到exp_dir目录
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    # 复制models/pointnet2_utils.py文件到exp_dir所代表的目录中
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    # 从已导入的MODEL模块中调用get_model方法获取一个分类器模型。
    # num_part可能是传递给get_model的一个参数，表示分类的部分或类别的数量。
    # normal_channel=args.normal是一个关键字参数，可能用于指定模型是否使用法线通道。
    # .cuda()方法将模型移动到GPU上，以便进行加速计算。
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    # 从MODEL模块中调用get_loss方法获取一个损失函数
    # 同样.cuda()方法将损失函数也移动到GPU上
    criterion = MODEL.get_loss().cuda()
    # 使用模型的apply方法对其所有层应用inplace_relu函数
    # 替换模型中的所有ReLU激活函数为in-place版本，以以节省GPU内存
    classifier.apply(inplace_relu)

    # 接受一个参数m，通常这个参数代表一个神经网络的层或模块
    def weights_init(m):
        # 获取m的类名，并将其赋值给classname
        classname = m.__class__.__name__
        # 检查classname是否包含字符串'Conv2d'，不包含返回-1。如果是，说明m是一个2D卷积层
        if classname.find('Conv2d') != -1:
            # torch.nn.init.xavier_normal_ 是一个权重初始化函数，也称为 Glorot 初始化
            # 它根据输入和输出单元的数量自动调整权重的初始值范围，以使得权重矩阵的初始值保持在一个合理的范围内，有助于模型更快地收敛
            # m.weight.data 表示 m 这个神经网络层的权重张量
            # xavier_normal_ 函数会直接修改这个张量的值，进行权重初始化
            torch.nn.init.xavier_normal_(m.weight.data)
            # torch.nn.init.constant_ 是一个偏置初始化函数，它将所有偏置初始化为给定的常数，这里是 0.0
            # m.bias.data 表示 m 这个神经网络层的偏置张量。constant_ 函数会直接修改这个张量的值，进行偏置初始化。
            torch.nn.init.constant_(m.bias.data, 0.0)
        # 检测m是否为全连接层
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        # 程序尝试加载一个之前保存的模型检查点
        # 当你保存了一个模型的检查点（checkpoint）后，这个方法允许你在后续的时间点从这些保存的参数中恢复模型的状态，以便继续训练或进行推理
        # exp_dir是一个路径，它指向保存检查点的目录
        # 程序将exp_dir转换为字符串，并拼接上'/checkpoints/best_model.pth'来得到完整的文件路径
        # 然后，它使用torch.load来加载该文件，并将结果保存在checkpoint变量中
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        # 从加载的检查点字典中，程序提取了epoch键对应的值，这个值表示模型训练到哪个周期（epoch）
        start_epoch = checkpoint['epoch']
        # 将保存在检查点中的模型参数加载到classifier模型中，使得模型可以从之前保存的状态继续训练或进行其他操作
        classifier.load_state_dict(checkpoint['model_state_dict'])
        # 程序记录了一条日志消息，表明它正在使用一个预训练的模型
        log_string('Use pretrain model')
    # 如果在try块中的代码执行期间发生任何异常，程序将跳转到这个except块
    except:
        # 在except块中，程序记录了一条日志消息，表明没有现有的模型，因此它将从头开始训练
        log_string('No existing model, starting training from scratch...')
        # 程序设置start_epoch为0，表示训练将从第一个周期开始
        start_epoch = 0
        # 程序使用weights_init函数来初始化classifier模型的权重
        # apply方法将weights_init函数应用于模型的每一个层
        classifier = classifier.apply(weights_init)

    # 程序检查命令行参数args中的optimizer键的值是否为'Adam'
    if args.optimizer == 'Adam':
        # 如果optimizer的值是'Adam'，则程序使用Adam优化器，并将模型的参数、学习率、betas参数、eps参数和权重衰减率传递给优化器
        optimizer = torch.optim.Adam(
            # 返回模型中所有可训练的参数（即权重和偏差）,这些参数将以迭代器的形式被优化器使用来更新模型的权重
            classifier.parameters(),
            # 这里设置了学习率（learning rate）。学习率是一个超参数，它控制模型参数在每次更新时的步长大小
            # 一个较小的学习率可能导致模型学习得更慢，而一个较大的学习率可能导致模型在最优解附近震荡甚至发散
            lr=args.learning_rate,
            # betas 参数是一个包含两个值的元组，分别对应于 Adam 算法中的一阶矩估计的指数衰减率和二阶矩估计的指数衰减率
            # 这两个超参数控制了动量（momentum）和速度（velocity）的衰减
            betas=(0.9, 0.999),
            #  这是一个非常小的数，用于防止除以零错误
            #  在 Adam 算法中，分母部分可能包含接近零的数，这可能导致数值不稳定
            #  通过设置 eps，我们可以确保分母不会变为零，从而保持数值稳定性。
            eps=1e-08,
            # weight_decay 是另一个正则化参数，它对应于 L2 正则化项
            # 通过在优化过程中向损失函数添加一个权重衰减项，有助于防止模型过拟合
            weight_decay=args.decay_rate
        )
    # 如果optimizer的值不是'Adam'，则执行以下代码
    else:
        # 程序使用随机梯度下降（SGD）优化器，并将模型的参数、学习率和动量传递给优化器
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    # 这个函数用于调整Batch Normalization层的动量值
    def bn_momentum_adjust(m, momentum):
        # 如果模块m是BatchNorm2d或BatchNorm1d类型，就将其动量设置为输入参数momentum的值
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    # 这里定义了学习率裁剪值、初始动量、动量衰减率和动量衰减步长
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    # 训练循环之前的准备
    # 初始化最佳准确率、全局epoch数、最佳类别平均IoU和最佳实例平均IoU
    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    # 这一行开始了一个主循环，它将遍历每个训练周期（epoch），从start_epoch开始，直到args.epoch
    for epoch in range(start_epoch, args.epoch):
        # 在每个epoch开始时，初始化一个空列表mean_correct，用于存储每个batch中正确预测的比例
        mean_correct = []

        # 输出当前epoch的信息到日志
        # global_epoch + 1: 全局的epoch计数，加1是为了使其从1开始，而不是从0。
        # epoch + 1: 当前的epoch计数，加1是为了从1开始计数。
        # args.epoch: 从命令行参数获取的总epoch数
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        # 计算新的学习率，它是初始学习率乘以衰减因子的一定幂次，但不能低于LEARNING_RATE_CLIP
        # args.learning_rate: 初始学习率。这是模型训练开始时使用的学习率
        # args.lr_decay: 学习率衰减率，它是一个小于1的数，用于在每次达到一定的训练轮次（epoch）后减小学习率
        # epoch: 当前的训练轮次。随着训练的进行，这个值会增加
        # args.step_size: 学习率衰减的步长。每隔args.step_size个训练轮次，学习率会乘以args.lr_decay进行衰减。
        # epoch // args.step_size: 这是一个整数除法操作，用于计算当前的训练轮次epoch已经经历了多少个完整的args.step_size
        # args.learning_rate * (args.lr_decay ** (epoch // args.step_size)): 这部分计算了经过多次衰减后的学习率。每次衰减都是将当前学习率乘以args.lr_decay
        # LEARNING_RATE_CLIP: 这是一个常数，用于限制学习率的最小值。如果经过衰减后的学习率低于这个值，那么实际使用的学习率将被设置为这个值
        # max(..., LEARNING_RATE_CLIP): 这部分确保了实际使用的学习率不会低于LEARNING_RATE_CLIP。如果计算出的学习率更低，那么将使用LEARNING_RATE_CLIP作为当前的学习率
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        # 输出新的学习率到日志
        log_string('Learning rate:%f' % lr)
        # 更新优化器中的学习率。
        # optimizer是一个PyTorch优化器实例
        # 在PyTorch中，优化器维护了一个或多个参数组（param_groups），每个参数组有自己的设置，如学习率、权重衰减等
        # 这段代码遍历所有参数组，并将每个参数组的学习率设置为新计算出的lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # 这段代码计算了Batch Normalization层应该使用的动量值
        # 与学习率类似，动量也根据训练轮次进行衰减，但有一个下限0.01。如果计算出的动量值低于这个下限，它将被设置为0.01。然后，代码打印出新的动量值。
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        # classifier是一个神经网络模型，并且该模型包含Batch Normalization层
        # classifier.apply()方法会对模型中的每个层应用bn_momentum_adjust函数（这个函数的具体实现没有给出，但应该是用来调整BN层的动量参数）
        # 通过这种方式，新的动量值被应用到模型的所有BN层
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        # 将classifier设置为训练模式
        # 在PyTorch中，某些层（如Dropout和Batch Normalization）在训练和评估时的行为是不同的，调用.train()确保这些层在训练时表现出正确的行为。
        classifier = classifier.train()

        '''learning one epoch'''
        # trainDataLoader是一个迭代器，通常用于加载训练数据
        # trainDataLoader会分批返回数据，每批数据包含points（点云数据）、label（标签）和target（可能是另一种形式的标签或目标值）
        # tqdm是一个进度条库，用于显示循环的进度
        # enumerate会返回每批数据的索引i和数据(points, label, target)，total=len(trainDataLoader)设置进度条的总长度，smoothing=0.9用于平滑进度条的更新。
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            # 在开始处理新的一批数据之前，需要将优化器中的梯度清零
            # 这是因为PyTorch在默认情况下会累积梯度，不清零的话，梯度会在每批数据上累积，导致不正确的更新
            optimizer.zero_grad()
            # 将points从PyTorch的Tensor对象转换为NumPy数组，以便进行后续的点云数据增强操作
            points = points.data.numpy()
            # 这两行代码对点云数据进行增强操作
            # random_scale_point_cloud函数对点云数据进行随机缩放，这有助于模型学习尺度不变的特征
            # shift_point_cloud函数对点云数据进行平移，这有助于模型学习位置不变的特征
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            # 将增强后的点云数据从NumPy数组转回PyTorch的Tensor对象，以便能够在GPU上进行计算
            points = torch.Tensor(points)
            # 使用float()方法将points的数据类型转换为浮点型
            # 使用long()方法将label和target的数据类型转换为长整型
            # 使用cuda()方法将所有数据移动到GPU上，以便进行加速计算
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            # 使用transpose方法调整points的维度顺序,以符合模型的输入要求
            points = points.transpose(2, 1)

            # classifier是一个神经网络模型，它接收points（点云数据）和经过to_categorical处理后的label（标签）作为输入
            # 这里假设to_categorical是一个将标签转换为one-hot编码的函数，num_classes表示类别的总数
            # seg_pred是模型的输出预测，可能表示每个点的分类结果
            # trans_feat可能是转换特征，用于后续操作，如损失计算
            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
            # 为了方便计算，先将seg_pred变为连续的内存块（如果它不是的话）
            # 通过view方法改变其形状,这里-1表示自动计算该维度的大小，以保持总元素数量不变，num_part可能是点的分组数量或部分数量
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            # 将target改变形状，使其成为一个列向量，然后通过索引[:, 0]将其转换为一维张量
            target = target.view(-1, 1)[:, 0]
            # 使用max函数在seg_pred的每一行上找到最大值的位置索引，这些索引即为预测的类别标签
            # 这里假设seg_pred的每一行对应一个点的分类分数
            pred_choice = seg_pred.data.max(1)[1]

            # 使用eq函数比较预测的类别标签pred_choice和目标标签target是否相等，返回一个布尔张量
            # 通过cpu()将其转移到CPU上（如果它之前在GPU上的话），并使用sum()计算正确预测的数量。
            correct = pred_choice.eq(target.data).cpu().sum()
            # 计算当前批次中正确预测的比例，并将其添加到mean_correct列表中
            # 这里假设args.batch_size是批次大小，args.npoint是每个批次中的点数
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            # 使用criterion计算损失。接收模型的预测seg_pred、目标标签target和转换特征trans_feat作为输入
            # seg_pred: 是一个张量（Tensor），其中包含了模型对每个点的分类预测。这些预测通常是概率分布的形式，表示每个点属于不同类别的可能性
            # target: 这是实际的目标标签，也就是我们希望模型能够学习并准确预测的真实值。target 可能是一个包含了每个点真实类别标签的张量
            # trans_feat: 特别是在处理点云数据时，可能会用到特征转换来提高模型的性能。这可以是对输入数据的某种预处理，也可以是在模型内部进行的一种特征空间的变换。
            loss = criterion(seg_pred, target, trans_feat)
            # 对损失进行反向传播，计算梯度
            loss.backward()
            # 使用优化器optimizer更新模型的参数
            optimizer.step()

        # 计算整个训练过程中的平均实例准确率
        train_instance_acc = np.mean(mean_correct)
        # 使用log_string函数输出训练准确率
        log_string('Train accuracy is: %.5f' % train_instance_acc)

        # 这行表示接下来的操作不需要计算梯度，常用于模型的测试或评估阶段
        with torch.no_grad():
            # 这行代码初始化了一个空字典test_metrics，用于存储测试过程中的各种指标或结果
            test_metrics = {}
            # 这里初始化了一个变量total_correct，用于记录在整个测试集中被正确分类或分割的样本总数
            total_correct = 0
            # total_seen变量用于记录测试过程中已经看过的样本总数
            total_seen = 0
            # 这行代码初始化了一个列表total_seen_class，长度为num_part，这个列表用于记录每个类别已经看过的样本数。
            total_seen_class = [0 for _ in range(num_part)]
            # 这行代码初始化了一个列表total_correct_class，用于记录每个类别中被正确分类或分割的样本数
            total_correct_class = [0 for _ in range(num_part)]
            # 这行代码初始化了一个字典shape_ious，它的键是seg_classes字典的键（即分类的类别）
            # 对于每个类别，它都初始化了一个空列表
            # 这个字典通常用于存储每个类别的Intersection over Union（IoU）值，这是评估分割任务性能的一个重要指标。
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            # 这行代码初始化了一个空字典seg_label_to_cat
            # 从注释中可以看出，这个字典可能用于将分割标签映射到相应的类别名（如“Airplane”或“Table”）
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            # 这段代码遍历了seg_classes字典的键（即类别名）和值（每个类别对应的标签列表）
            # seg_classes是一个形如{"Airplane": [0, 1], "Car": [2, 3], ...}的字典，其中每个类别（如"Airplane"）映射到一个或多个标签（如[0, 1]）。
            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    # 代码将每个 label 作为键，将相应的 cat 作为值，添加到 seg_label_to_cat 字典中
                    # {"Airplane": [0], "Car": [1]}，那么 seg_label_to_cat 将被构建为 {0: "Airplane", 1: "Car"}。
                    seg_label_to_cat[label] = cat

            # 在PyTorch等深度学习框架中，eval()方法通常用于将模型设置为评估模式
            # 在评估模式下，某些特定于训练的层（如dropout和batch normalization）将改变它们的行为
            # 此外，这也是一个信号，表明接下来的代码将使用模型进行预测，而不是训练
            classifier = classifier.eval()

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                # 获取当前批次的大小cur_batch_size和每个点云中的点数NUM_POINT
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                # to_categorical(label, num_classes): 用于将整数标签转换为独热编码（one-hot encoding）
                # label包含了每个点云样本的类别标签，num_classes是类别的总数
                # classifier(points, ...): 使用分类器`classifier`对点云`points`进行预测。返回的`seg_pred`是部件分割的预测结果。
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                # 将预测结果从GPU移动到CPU,并将PyTorch张量转换为NumPy数组
                cur_pred_val = seg_pred.cpu().data.numpy()
                # 保存原始的预测逻辑值（logits）
                # 在深度学习中，逻辑值（logits）通常指的是模型在最后一层（通常是softmax层之前）的输出值
                # 这些值还没有经过softmax函数或其他激活函数的处理，因此它们还没有被转换成概率分布
                # 在某些情况下，保存这些原始的logits值是有用的，因为它们提供了比概率更丰富的信息，特别是在计算某些类型的损失函数或进行其他形式的模型分析时
                cur_pred_val_logits = cur_pred_val
                # 建一个全零数组，用于后续存储最终的预测结果
                # 这里使用`np.int32`数据类型来确保可以存储整数标签
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                # cur_batch_size表示当前批次中的样本数（或点云数量）
                for i in range(cur_batch_size):
                    # target[i, 0]：从target数组中取出第i个样本的类别标签。因此，target[i, 0]就是第i个样本点的真实部件标签。
                    # seg_label_to_cat：这是一个映射，它将部件标签映射到另一个表示或索引。这个映射是为了将原始标签转换为模型内部使用的标签格式，或者是为了方便处理不同数量的部件类别。
                    # cat：存储了映射后的结果，它可能是一个索引或一个类别组标识符，用于后续从logits中选择相关的部分。
                    cat = seg_label_to_cat[target[i, 0]]
                    # cur_pred_val_logits是一个三维数组，包含了模型对每个样本点的原始预测输出（logits）
                    # 这里选取了第i个样本的所有点的logits。其中，第一维是样本索引，第二维是点的索引，第三维是每个点的logits值（对应不同的部件类别）
                    logits = cur_pred_val_logits[i, :, :]
                    # logits[:, seg_classes[cat]]：这里首先从logits中选取与当前部件类别cat相关的部分
                    # seg_classes可能是一个映射,将部件类别映射到它们在logits中的列索引或范围
                    # np.argmax(..., 1)：这个函数沿着第二个维度（即点的维度）找出最大logits值的索引。这个索引代表了模型预测的部件类别，因为argmax函数返回的是得分最高的类别的索引
                    # ... + seg_classes[cat][0]：这里对找出的索引进行偏移，以得到最终的部件标签。
                    # 偏移量seg_classes[cat][0]给出了当前部件类别在标签空间中的起始索引,因为不同的部件类别可能有不同的标签范围，而模型的输出logits通常是连续的，没有预留间隔。
                    # cur_pred_val[i, :]：将计算出的预测标签存储到cur_pred_val数组的相应位置。
                    # cur_pred_val用于存储当前批次中所有样本的预测结果，其形状与target相同，以便于后续的性能比较。
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                # cur_pred_val == target：这是一个逐元素的比较操作，它比较cur_pred_val数组（模型的预测标签）和target数组（真实的标签）中的每个元素是否相等。
                # 结果是一个布尔数组，其形状与cur_pred_val和target相同，且每个位置上的值为True或False，表示该位置的预测是否正确。
                # np.sum(...)：对布尔数组求和。
                # 在NumPy中，True被当作1，False被当作0。因此，求和操作实际上计算了数组中True的数量，即预测正确的点的总数。
                # correct：存储了当前批次中预测正确的点的数量。
                correct = np.sum(cur_pred_val == target)
                # total_correct：是一个累积变量，用于跟踪到目前为止所有批次中预测正确的点的总数。
                total_correct += correct
                # cur_batch_size：当前批次中的样本数（或点云数）。
                # NUM_POINT：每个样本中的点数。这是一个常量，表示每个点云包含多少个点。
                # cur_batch_size * NUM_POINT：计算了当前批次中总共包含多少个点。
                # total_seen：是一个累积变量，用于跟踪到目前为止模型已经处理过的点的总数。
                total_seen += (cur_batch_size * NUM_POINT)

                # 这段代码遍历了所有的部件类别
                # total_seen_class[l]统计了到目前为止，真实标签中属于类别l的点的总数
                # total_correct_class[l]统计了到目前为止，模型正确预测为类别l的点的总数，通过比较cur_pred_val（模型的预测）和target（真实标签）来确定哪些点的预测是正确的
                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                # 这段代码的主要目的是计算每个形状的部件级别的IoU（交并比），并将其存储在shape_ious字典中，以便后续分析模型在点云部件分割任务上的性能
                for i in range(cur_batch_size):
                    # 对于每个样本，从cur_pred_val数组中提取模型的预测标签（segp），从target数组中提取真实的标签（segl）
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    # 使用真实标签数组的第一个元素（segl[0]）来查找它所属的大类别（cat）
                    # 这里假设seg_label_to_cat是一个字典，它将标签映射到对应的大类别
                    cat = seg_label_to_cat[segl[0]]
                    # 对于当前样本所属的大类别，初始化一个与该类别下部件数量相等的列表part_ious，用于存储每个部件的IoU值。列表中的每个元素初始值设为0.0。
                    # 这个列表推导式生成的结果（即一个长度为len(seg_classes[cat])，所有元素都是0.0的列表）被赋值给变量part_ious。这个列表稍后将用于存储每个部件类别的IoU（交并比）值
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    # 计算每个部件（part）类别的交并比（Intersection over Union，IoU）
                    # 遍历seg_classes[cat]中的每个元素，其中seg_classes是一个字典，将大类别（如“椅子”、“汽车”等）映射到它们各自的部件类别列
                    # 变量cat表示当前样本所属的大类别，而l则代表这个大类别下的一个具体部件类别
                    for l in seg_classes[cat]:
                        # np.sum(segl == l) == 0：在真实标签segl中，部件类别l的点的数量是否为0。换句话说，它检查当前样本中是否不存在部件l。
                        # np.sum(segp == l) == 0：在预测标签segp中，部件类别l的点的数量是否为0。这意味着模型没有预测出部件l。
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            # part_ious是一个列表，用于存储当前大类别下每个部件的IoU值
                            # 由于列表索引通常从0开始，而部件类别l可能不是从0开始的整数，因此代码使用l - seg_classes[cat][0]来计算正确的索引。
                            # 在这种情况下，由于真实和预测中都没有部件l，所以将其IoU值设为1.0，表示没有错误（或者说，模型正确地预测了这个部件不存在）。
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            # 分子np.sum((segl == l) & (segp == l))计算了真实和预测中都为部件l的点的数量（交集）
                            # 分母np.sum((segl == l) | (segp == l))计算了真实或预测中为部件l的点的数量（并集）
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    # shape_ious[cat]：这部分代码访问了字典 shape_ious 中键为 cat 的项。
                    # 由于 shape_ious 的目的是存储每个大类别的平均 IoU，因此 shape_ious[cat] 应该是一个列表，用于存储多次评估或不同样本中当前大类别的平均 IoU 值。
                    # np.mean(part_ious)：计算 part_ious 列表中所有元素的平均值。
                    #
                    # .append(...)：这个方法将计算得到的平均 IoU 值添加到 shape_ious[cat] 列表的末尾。这意味着每次评估一个样本时，都会将其平均部件 IoU 值添加到对应大类别的列表中。
                    shape_ious[cat].append(np.mean(part_ious))

            # 初始化了一个空列表 all_shape_ious，用于存储所有大类别在所有样本上的IoU值
            all_shape_ious = []
            # 在这个双层循环中，外层循环遍历 shape_ious 字典中的所有大类别（键）
            # 内层循环遍历每个大类别下的 IoU 值列表，并将这些 IoU 值添加到 all_shape_ious 列表中
            # 然后，使用 NumPy 的 mean 函数计算每个大类别 IoU 值的平均值，并将这个平均值更新为 shape_ious 字典中该大类别的值。
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            # 这行代码将 shape_ious 字典中的所有值（即每个大类别的平均 IoU）转换为一个列表，并计算这些平均 IoU 值的总体平均值
            mean_shape_ious = np.mean(list(shape_ious.values()))
            # 这行代码计算了测试集上的总体准确率，其中 total_correct 是正确预测的点的总数，total_seen 是测试集中所有点的总数
            # 这个准确率指标衡量了模型在测试集上的整体性能
            test_metrics['accuracy'] = total_correct / float(total_seen)
            # 这行代码计算了每个大类别的平均准确率
            # 假设 total_correct_class 和 total_seen_class 是两个列表，分别存储了每个大类别正确预测的点的数量和每个大类别中所有点的数量
            # 这里，每个大类别的准确率被计算出来，然后对这些准确率求平均，得到所有大类别的平均准确率的总体平均值。
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
            # 在这个循环中，代码遍历了 shape_ious 字典中的所有大类别（键），这些键经过排序以确保输出的一致性。
            for cat in sorted(shape_ious.keys()):
                # 字符串格式化用于确保每个大类别的名称都占据相同的空间，从而使输出更加整齐。
                # 对于每个大类别，使用 log_string 函数打印该类别的名称和平均 IoU 值
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            # 这两行代码将之前计算得到的评估指标记录到 test_metrics 字典中
            # 第一行记录了所有大类别的平均 IoU 的总体平均值，使用键 'class_avg_iou'
            # 第二行记录了所有大类别在所有样本上的 IoU 值的总体平均值
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        # 这行代码使用 log_string 函数打印当前周期（epoch）的测试准确率、类别平均 mIOU（mean Intersection over Union）以及实例平均 mIOU
        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
        # 这行代码检查当前周期的实例平均 mIOU 是否大于或等于之前记录的最佳实例平均 mIOU（best_inctance_avg_iou）。如果是，则进入代码块来保存当前模型状态
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            # 使用 logger.info 记录信息，表明正在保存模型
            logger.info('Save model...')
            # 确定模型保存的路径，并使用 log_string 记录该路径
            # 这里，checkpoints_dir 是一个指向检查点目录的路径对象或字符串，模型将保存在这个目录下的 'best_model.pth' 文件中。
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            # 创建一个字典 state，其中包含当前周期的编号、训练准确率、测试准确率、类别平均 mIOU、实例平均 mIOU、
            # 模型的状态字典（classifier 是模型对象）和优化器的状态字典（optimizer.state_dict()）
            # 这些信息对于之后恢复模型训练或进行模型分析非常有用
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        #
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        log_string('Best accuracy is: %.5f' % best_acc)
        log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
