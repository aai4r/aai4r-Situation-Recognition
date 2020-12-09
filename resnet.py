import math
import torch.nn as nn
import torchvision as tv


def initLinear(linear, val=None):
    if val is None:
        fan = linear.in_features + linear.out_features
        spread = math.sqrt(2.0) * math.sqrt(2.0 / fan)
    else:
        spread = val
    linear.weight.data.uniform_(-spread, spread)
    linear.bias.data.uniform_(-spread, spread)


class VGGModified(nn.Module):
    def __init__(self):
        super(VGGModified, self).__init__()
        self.vgg = tv.models.vgg16(pretrained=True)
        self.vgg_features = self.vgg.features
        # self.classifier = nn.Sequential(
        # nn.Dropout(),
        self.lin1 = nn.Linear(512 * 7 * 7, 1024)
        self.relu1 = nn.ReLU(True)
        self.dropout1 = nn.Dropout()
        self.lin2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU(True)
        self.dropout2 = nn.Dropout()

        initLinear(self.lin1)
        initLinear(self.lin2)

    def rep_size(self): return 1024

    def forward(self, x):
        return self.dropout2(self.relu2(self.lin2(self.dropout1(self.relu1(self.lin1(self.vgg_features(x).view(-1, 512*7*7)))))))


class ResNetModifiedLarge(nn.Module):
    def __init__(self):
        super(ResNetModifiedLarge, self).__init__()
        self.resnet = tv.models.resnet101(pretrained=True)
        # probably want linear, relu, dropout
        self.linear = nn.Linear(7*7*2048, 1024)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        initLinear(self.linear)

    def base_size(self): return 2048
    def rep_size(self): return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.dropout2d(x)

        # print x.size()
        return self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))


class ResNetModifiedMedium(nn.Module):
    def __init__(self):
        super(ResNetModifiedMedium, self).__init__()
        self.resnet = tv.models.resnet50(pretrained=True)
        # probably want linear, relu, dropout
        self.linear = nn.Linear(7*7*2048, 1024)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        initLinear(self.linear)

    def base_size(self): return 2048
    def rep_size(self): return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.dropout2d(x)

        # print x.size()
        return self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))


class ResNetModifiedSmall(nn.Module):
    def __init__(self):
        super(ResNetModifiedSmall, self).__init__()
        self.resnet = tv.models.resnet34(pretrained=True)
        # probably want linear, relu, dropout
        self.linear = nn.Linear(7*7*512, 1024)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        initLinear(self.linear)

    def base_size(self): return 512
    def rep_size(self): return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.dropout2d(x)

        return self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))
