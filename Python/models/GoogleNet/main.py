import torch
import torch.nn as nn

class Inception(nn.Module):
    def __init__(self, in_channels, f1x1, f3x3red, f3x3, f5x5red, f5x5, pool_proj):
        """
        :param in_channels: 输入数据的通道数
        :param f1x1: 1 * 1卷积层的输出通道数
        :param f3x3red: 3 * 3卷积层中1 * 1 卷积层的输出通道数
        :param f3x3: 3 * 3 卷积层的输出通道数
        :param f5x5red: 5 * 5卷积层中1 * 1 卷积层的输出通道数
        :param f5x5: 5 * 5卷积层的输出通道数
        :param pool_proj:池化层中1 * 1 卷积层的输出通道数
        """
        super(Inception, self).__init__()

        # 1x1 conv branch
        """Sequential
        下面的这段代码是一个包含三个操作的序列（Sequential）模块，用于对输入进行卷积操作。
        具体解释如下：
        1.nn.Conv2d(in_channels, f1x1, kernel_size=1),这个的意思是——这是一个二维卷积层，
          输入通道数为in_channels，输出通道数为f1 * 1，卷积核大小为1 * 1,意味着对每个像素点进行
          一次通道上的线性变换。
        2.nn.BatchNorm2d(f1x1),这算是一个二维批量归一化层，对卷积输出进行归一化处理，使得每个特
          征的均值为0、方差为1，加速网络的训练收敛。
        3.nn.ReLU(inplace=True)这是一个ReLU激活函数，将输入中所有小于0的值都置为0，保留所有≥0
          的值，防止梯度消失问题。inplace = True表示将原地修改输入，节省显存空间。
         因此，整个模块的作用是对输入进行 1 * 1卷操作，接着批量归一化和ReLU激活函数处理，输出特征图。 
        """

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, f1x1, kernel_size=1),
            nn.BatchNorm2d(f1x1),
            nn.ReLU(inplace=True)
        )
        """"
        这段代码定义了一个包含两个卷积层的神经网络模块。主要包含以下几个部分：
        1.nn.Conv2d(in_channels, f3x3red, kernel_size=1)：定义了一个1 * 1 的卷积层，其中
          in_channels表示输入数据的通道数，f3x3red表示输出数据的通道数，kernel_size表示卷积核的大小。
        2.nn.BatchNorm2d(f3x3red)：对输出数据进行批量标准化操作，加速收敛和提高模型性能。
        3.nn.ReLU(inplace=True),对输出数据进行ReLU激活函数操作，增强模型的非线性表达能力
        4.nn.Conv2d(f3x3red, f3x3, kernel_size=3, padding=1),定义了一个3 * 3的卷积层，其中f3x3red表示输入数
          据的通道数，f3x3表示输出数据的通道数，kernel_size表示卷积核的大小，padding=1表示在卷积操作中对输入数据进行1像素的零填充。
        5.nn.BatchNorm2d(f3x3),对输出数据进行批量标准化操作。
        6.nn.ReLU(inplace=True)对输出数据进行ReLU激活函数操作。
        整个网络模块的作用是将输入数据经过1 * 1 的卷积层，将通道数压缩，在经过一个3 * 3 的卷积层进行特征提取。这样做的好处是可以减少参数数量，
        加速计算，同时增强模型的表达能力，提高模型的性能。 
        """
        # 3x3 conv branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, f3x3red, kernel_size=1),
            nn.BatchNorm2d(f3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(f3x3red, f3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(f3x3),
            nn.ReLU(inplace=True)
        )


        """"
        这是一个卷积神经网络的分枝，用于提取图像特征。具体来说，它包含了以下层：
        1.nn.Conv2d(in_channels, f5x5red, kernel_size=1),这是一个1 * 1 的卷积层，用于减少通道数(in_channels)至f5x5red。这个操作有助于降低计算量并调高模型效果。
        2.nn.BatchNorm2d(f5x5red)：这是一个批量归一化层用于规范化输入数据，加速收敛并防止过拟合。
        3.nn.ReLU(inplace=True)：这是一个ReLU激活函数，用于引入非线性特征，让模型能够学习更加复杂的特征。
        4.nn.Conv2d(f5x5red, f5x5, kernel_size=5, padding=2)：这是一个5 * 5 的卷积层，用于提取更大范围的特征，padding=2表示 在输入图像周围添加一圈0，保证输出图像大小与输入图像相同。
        5.nn.BatchNorm2d(f5x5)：这是一个批量归一化层，同样用于规范化输入数据。
        6.nn.ReLU(inplace=True)：这是另一个ReLU激活函数，用于引入非线性特征。
        这些层组成了一个卷积神经网络分支，可以用于提取输入图像中的特征。
        """
        # 5x5 conv branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, f5x5red, kernel_size=1),
            nn.BatchNorm2d(f5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(f5x5red, f5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(f5x5),
            nn.ReLU(inplace=True)
        )



        """
        这是一个在神经网络中使用的卷积神经网络（CNN）模块，包含以下四个层：
        1.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),是一个最大池化层，使用大小为3x3的滤波器在输入上进行池化操作，并且步长为1，填充为1。池化层是CNN中的一种常见操作，
          它可以减少特征映射的维数，从而提高计算效率，并且有助于提取图像特征。
        2.nn.Conv2d(in_channels, pool_proj, kernel_size=1),这是一个卷积层，使用大小为1x1的滤波器在经过池化层处理后的输入上进行卷积操作。这里的 in_channels 是输入通道数，
          pool_proj 是输出通道数，表示卷积操作的输出特征映射的数量。卷积层在CNN中也是一种常见操作，它可以将输入特征映射转换为输出特征映射，并且有助于提取图像特征。
        3.nn.BatchNorm2d(pool_proj),这是一个批量归一化层，用于对卷积层的输出进行归一化处理，加速模型的训练，并且有助于防止过拟合。
        4.nn.ReLU(inplace=True)这是一个激活函数层，使用ReLU激活函数对归一化后的输出进行非线性变换，增强模型的表达能力。其中 inplace=True 表示将原地修改输入数据，节省内存开销。
        """
        # max pooling branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        output = torch.cat([x1, x2, x3, x4], dim=1)
        return output

class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogleNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
