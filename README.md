基于PyTorch和CIFAR-10的图像分类算法
====

*各个模型的准确度*

| model        | epoch=10  | epoch=20  | epoch=30  |
| :----------: | :-------: | :-------: | :-------: |
| resnet18     | 0.7737    | 0.7794    | 0.7857    |
| resnet34     | 0.7655    | 0.7849    | 0.7867    |
| resnet50     | 0.6766    | 0.7769    | 0.7786    |
| resnet101    | 0.4119    | 0.3903    | 0.5182    |

> 20个epoch之后，过拟合越来越严重

*一步步优化resnet18*

优化数据增强，`0.7857->0.8174`，如下：
```
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
```

添加权重约束，30个EPOCH后，训练集上准确率由`0.8174`下降到`0.77414`，但是过拟合程度大大降低，可以继续训练，如下：
```
optimizer = torch.optim.Adam(net.parameters(), LR, weight_decay=5e-4)
```

使用SGD优化器，`lr`为`0.001`，30个EPOCH后训练集上的准确率为`0.77420`，如下：
```
optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
```

使用学习率衰减策略，初始学习率为`0.1`，经过500个EPOCH后在训练集上的准确率为`0.8432`，如下：
```
optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=0.1)
```

**参考资料**
* 权重衰减与学习率衰减:https://blog.csdn.net/program_developer/article/details/80867468
* Pytorch中的学习率衰减及其用法:https://www.jianshu.com/p/26a7dbc15246