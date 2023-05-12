# Resnet-34

## Resnet-34 section code comments

```import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, dim, out_dim, stride) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(dim, out_dim, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)  # 一个卷积层，它应用于输入 x,用于提取特征。
        x = self.bn1(x)  # 是批量归一化层，加速收敛并提高性能。
        x = self.relu1(x)  # 激活函数，用于在神经网络中引入非线性，能够加速训练并提高模型性能。
        x = self.conv2(x)  # 一个卷积层，它应用于输入x,用于提取特征。
        x = self.bn2(x)  # 是批量归一化层，加速收敛并提高性能。
        x = self.relu2(x)  # 激活函数，用于在神经网络中引入非线性，能够加速训练并提高模型性能。
        return x


class ResNet32(nn.Module):
    def __init__(self, in_channel=64, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = in_channel

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3)
        self.maxpooling = nn.MaxPool2d(kernel_size=2)
        self.last_channel = in_channel

        self.layer1 = self._make_layer(in_channel=64, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(in_channel=128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(in_channel=256, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(in_channel=512, num_blocks=3, stride=2)

        self.avgpooling = nn.AvgPool2d(kernel_size=2)
        self.classifier = nn.Linear(4608, self.num_classes)

    def _make_layer(self, in_channel, num_blocks, stride):
        layer_list = [Block(self.last_channel, in_channel, stride)]
        self.last_channel = in_channel
        for i in range(1, num_blocks):
            b = Block(in_channel, in_channel, stride=1)
            layer_list.append(b)
        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.conv1(x)  # [bs, 64, 56, 56] 特征提取过程
        x = self.maxpooling(x)  # [bs, 64, 28, 28]池化，降低分辨率和计算量
        x = self.layer1(x)  # [bs, 64, 56, 56] 残差层，是一个由3个Block组成的序列，对输入数据进行特征提取。
        x = self.layer2(x)  # [bs, 128, 28, 28] 残差层，是一个由4个Block组成的序列，对输入数据进行特征提取。
        x = self.layer3(x)  # [bs, 256, 14, 14] 残差层，是一个由6个Block组成的序列，对输入数据进行特征提取。
        x = self.layer4(x)  # [bs, 512, 7, 7] 残差层，是一个由3个Block组成的序列，对输入数据进行特征提取。
        x = self.avgpooling(x)  # [bs, 512, 3, 3]对输入数据x进行平均池化，降低数据维度，去除冗余信息。
        x = x.view(x.shape[0], -1)  # [bs, 4608]将张量 x 重新调整为一个二维张量，
        x = self.classifier(x)  # [bs, self.num_classes]将输入数据x通过一个全连接层进行分类。
        output = F.softmax(x)  # [bs, self.num_classes]对输入数据x进行softmax操作,每一行都被转换为概率分布

        return output


if __name__=='__main__':
    t = torch.randn([8, 3, 224, 224])
    model = ResNet32()
    out = model(t)
    print(out.shape)
```
