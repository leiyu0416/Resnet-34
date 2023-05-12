# Resnet-34

## What grammar issues occurred in the template for building the model and what consequences would it have?

```import torch 
 import torch . nn as nn 
 import torch . nn . functional as F 

class xxxNet ( nn . Module ):
 def __ init __( self ):
 pass 
 
 def forward ( x ):
 return x 
```
### The problem with the previous code segment

   If the parent class nn.Module is not called__ init__ Method, then the xxxNet class will not inherit the properties and methods of the nn.Module class. In this way, the xxxNet class cannot use the methods provided by the nn.Module class, nor can it perform model training and evaluation correctly, which will result in the model not working properly.So, we need to call the parent class's__ init__ method.

   By using self, we can access and modify the properties and methods of this instance object in the methods of the class.

### The code modification is as follows

```import torch
import torch.nn as nn
import torch.nn.functional as F

class xxxNet(nn.Module):
    def __init__(self):
        super(xxxNet, self).__init__() # 调用父类的__init__的方法
    
    def forward(self, x):
        return x
 ```

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
        x = self.conv1(x)  # [bs, 64, 112, 112] 特征提取过程
        x = self.maxpooling(x)  # [bs, 64, 56, 56]池化，降低分辨率和计算量
        x = self.layer1(x)  # [bs, 64, 56, 56] 残差层，是一个由3个Block组成的序列，对输入数据进行特征提取。
        x = self.layer2(x)  # [bs, 128, 28, 28] 残差层，是一个由4个Block组成的序列，对输入数据进行特征提取。
        x = self.layer3(x)  # [bs, 256, 14, 14] 残差层，是一个由6个Block组成的序列，对输入数据进行特征提取。
        x = self.layer4(x)  # [bs, 512, 7, 7] 残差层，是一个由3个Block组成的序列，对输入数据进行特征提取。
        x = self.avgpooling(x)  # [bs, 512, 3, 3] 对输入数据x进行平均池化，降低数据维度，去除冗余信息。
        x = x.view(x.shape[0], -1)  # [bs, 4608] 将张量 x 重新调整为一个二维张量，-1表示自动计算该维度的大小，使得张量的总大小保持不变。
        x = self.classifier(x)  # [bs, num_classes] 将输入数据x通过一个全连接层进行分类。
        output = F.softmax(x)  # [bs, num_classes] 对输入数据x进行softmax操作,每一行都被转换为概率分布

        return output


if __name__=='__main__':
    t = torch.randn([8, 3, 224, 224])
    model = ResNet32()
    out = model(t)
    print(out.shape)
```

## Functions of each parameter

kernel_ Size=3 indicates that the size of the convolutional kernel is 3x3.

Street=street indicates that the step size of each movement of the convolutional kernel is street.

Padding=1 indicates filling 1 pixel at the edge of the image before the convolution operation.

Conv1: A convolutional layer with a convolutional kernel size of 7x7, a step size of 2, and a padding of 3.

Bn1: A batch normalization layer. It can accelerate the training of the model and improve its accuracy.

Relu1: a activation function layer, which is activated using the ReLU function. It can make the model more sparse, reduce the number of parameters, and thus improve the generalization ability of the model.

Maxpooling: A pooling layer that uses maximum pooling for downsampling. Maximizing pooling is a common pooling method, which can reduce the size of the feature map, improve the calculation efficiency, and reduce the risk of overfitting.

Layer1: is the first residual block in the ResNet32 model, which accelerates model training and improves model accuracy

Avgpooling: It is a pooling layer in the ResNet32 model that uses adaptive average pooling for global pooling. Adaptive average pooling is a commonly used pooling method that can adaptively perform pooling operations based on the size of the input feature map, thereby obtaining a fixed size output feature map.

Self. classifier: A fully connected layer used to map the output feature maps of convolutional layers to category labels.
