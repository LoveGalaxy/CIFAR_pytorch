import torch
import torch.nn as nn

class Bottleneck(nn.Module):

    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, 1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.pool(self.conv(self.relu(self.bn(x))))
        return out

class DenseNet(nn.Module):
    
    def __init__(self, block, nblocks, num_planes=64, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        self.conv1 = nn.Conv2d(3, num_planes, 7, 1, 3, bias=False)
        # self.bn1 = BatchNorm2d(num_planes)
        # self.relu = nn.ReLU(inplace=True)
        # self.pool = nn.MaxPool2d(3, 2, 1)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        in_planes, num_planes = self._compute_planes(num_planes, growth_rate, nblocks[0], reduction)
        self.trans1 = Transition(in_planes, num_planes)

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        in_planes, num_planes = self._compute_planes(num_planes, growth_rate, nblocks[1], reduction)
        self.trans2 = Transition(in_planes, num_planes)

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        in_planes, num_planes = self._compute_planes(num_planes, growth_rate, nblocks[2], reduction)
        self.trans3 = Transition(in_planes, num_planes)

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        in_planes, num_planes = self._compute_planes(num_planes, growth_rate, nblocks[3], reduction)

        self.bn_o = nn.BatchNorm2d(in_planes)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(in_planes, num_classes)

    def _compute_planes(self, in_planes, growth_rate, nblock, reduction):
        in_planes = nblock * growth_rate + in_planes
        out_planes = int(in_planes * reduction)
        return in_planes, out_planes


    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.avgpool(self.bn_o(self.dense4(out)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def DenseNet121(num_classes):
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_classes=num_classes)

def DenseNet169(num_classes):
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32, num_classes=num_classes)

def DenseNet201(num_classes):
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32, num_classes=num_classes)

def DenseNet161(num_classes):
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48, num_classes=num_classes)

def DenseNet_CIFAR(num_classes):
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12, num_classes=num_classes)


if __name__ == "__main__":
    net = DenseNet121(10)
    x = torch.randn(1, 3, 32, 32)
    print(net)
    y = net(x)
    print(y.shape)