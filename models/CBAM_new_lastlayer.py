# ---------------------------------------------------- #
# CBAM注意力机制
# 结合了通道注意力机制和空间注意力机制
# ---------------------------------------------------- #

import torch
from torch import nn


# （1）通道注意力机制
class channel_attention(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接的通道下降倍数
    def __init__(self, in_channel, ratio=4):
        # 继承父类初始化方法
        super(channel_attention, self).__init__()

        # 全局最大池化 [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 第一个全连接层, 通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # 第二个全连接层, 恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        # relu激活函数
        self.relu = nn.ReLU()
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 获取输入特征图的shape
        b, c, h, w = inputs.shape

        # 输入图像做全局最大池化 [b,c,h,w]==>[b,c,1,1]
        max_pool = self.max_pool(inputs)
        # 输入图像的全局平均池化 [b,c,h,w]==>[b,c,1,1]
        avg_pool = self.avg_pool(inputs)

        # 调整池化结果的维度 [b,c,1,1]==>[b,c]
        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])

        # 第一个全连接层下降通道数 [b,c]==>[b,c//4]
        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        # 激活函数
        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        # 第二个全连接层恢复通道数 [b,c//4]==>[b,c]
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        # 将这两种池化结果相加 [b,c]==>[b,c]
        x = x_maxpool + x_avgpool
        # sigmoid函数权值归一化
        x = self.sigmoid(x)
        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])
        # 输入特征图和通道权重相乘 [b,c,h,w]
        outputs = inputs * x

        return outputs


# ---------------------------------------------------- #
# （2）空间注意力机制
class spatial_attention(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, kernel_size=7):
        # 继承父类初始化方法
        super(spatial_attention, self).__init__()

        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)

        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)

        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)
        # 空间权重归一化
        x = self.sigmoid(x)
        # 输入特征图和空间权重相乘
        outputs = inputs * x

        return outputs


# ---------------------------------------------------- #
# （3）CBAM注意力机制
class cbam(nn.Module):
    # 初始化，in_channel和ratio=4代表通道注意力机制的输入通道数和第一个全连接下降的通道数
    # kernel_size代表空间注意力机制的卷积核大小
    def __init__(self, in_channel, ratio=4, kernel_size=7):
        # 继承父类初始化方法
        super(cbam, self).__init__()

        # 实例化通道注意力机制
        self.channel_attention = channel_attention(in_channel=in_channel, ratio=ratio)
        # 实例化空间注意力机制
        self.spatial_attention = spatial_attention(kernel_size=kernel_size)

    # 前向传播
    def forward(self, inputs):
        # 先将输入图像经过通道注意力机制
        x = self.channel_attention(inputs)
        # 然后经过空间注意力机制
        x = self.spatial_attention(x)

        return x


class dehaze_net(nn.Module):

    def __init__(self):
        super(dehaze_net, self).__init__()
        self.attention = cbam(3)  # 3为输入的通道数目
        self.relu = nn.ReLU(inplace=True)

        # 参数含义 输入通道数，输出通道数，卷积核大小，卷积步幅，填充添加到输入的所有四个边,为输出添加可学习的偏差
        self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(3, 32, 1, 1, 0, bias=True)  # 进行1x1x32的卷积
        self.s1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 3x3最大池化

        # concat conv2 pool1
        self.e_conv3 = nn.Conv2d(64, 32, 1, 1, 0, bias=True)
        # 3次连续卷积，用于提取特征
        self.e_conv4 = nn.Conv2d(32, 64, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        self.s2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)  # 5x5最大池化

        # concat pool2 和conv6 输入conv7
        self.e_conv7 = nn.Conv2d(64, 32, 1, 1, 0, bias=True)
        self.s3 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)  # 7x7最大池化

        # concat pool3 和conv7 输入conv8
        self.e_conv8 = nn.Conv2d(64, 3, 1, 1, 0, bias=True)

    def forward(self, x):
        source = []
        source.append(x)

        x1 = self.relu(self.e_conv1(x))
        #x1 = self.attention(x0)  # 添加注意力机制

        x2 = self.relu(self.e_conv2(x1))
        x3 = self.s1(x2)

        concat1 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.e_conv3(concat1))
        x5 = self.relu(self.e_conv4(x4))
        x6 = self.relu(self.e_conv5(x5))
        x7 = self.relu(self.e_conv6(x6))
        x8 = self.s1(x7)

        concat2 = torch.cat((x7, x8), 1)
        x9 = self.relu(self.e_conv7(concat2))
        x10 = self.s3(x9)

        concat3 = torch.cat((x10, x9), 1)
        x11 = self.attention(concat3)
        x12 = self.relu(self.e_conv8(x11))

        clean_image = self.relu((x12 * x) - x12 + 1)

        return clean_image


'''import torch
import torch.nn as nn
import math

class dehaze_net(nn.Module):

	def __init__(self):
		super(dehaze_net, self).__init__()

		self.relu = nn.ReLU(inplace=True)

		#参数含义 输入通道数，输出通道数，卷积核大小，卷积步幅，填充添加到输入的所有四个边,为输出添加可学习的偏差
		self.e_conv1 = nn.Conv2d(3,3,1,1,0,bias=True)
		self.e_conv2 = nn.Conv2d(3,3,3,1,1,bias=True) 
		self.e_conv3 = nn.Conv2d(6,3,5,1,2,bias=True) 
		self.e_conv4 = nn.Conv2d(6,3,7,1,3,bias=True) 
		self.e_conv5 = nn.Conv2d(12,3,3,1,1,bias=True)


	def forward(self, x):
		source = []
		source.append(x)

		x1 = self.relu(self.e_conv1(x))
		x2 = self.relu(self.e_conv2(x1))

		concat1 = torch.cat((x1,x2), 1)
		x3 = self.relu(self.e_conv3(concat1))

		concat2 = torch.cat((x2, x3), 1)
		x4 = self.relu(self.e_conv4(concat2))

		concat3 = torch.cat((x1,x2,x3,x4),1)
		x5 = self.relu(self.e_conv5(concat3))

		clean_image = self.relu((x5 * x) - x5 + 1) 

		return clean_image



'''
