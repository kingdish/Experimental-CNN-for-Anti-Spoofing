import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision import models
import os
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyNetv1(nn.Module):
    def __init__(self):
        super(MyNetv1, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, 5)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.bn3 = nn.BatchNorm2d(32)
        # self.fc1 = nn.Linear(16*10*10, 256)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        # out = F.relu(self.conv1(x))
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        # out = F.relu(self.conv2(out))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.avg_pool2d(out, 16)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc2(out)
        return out


class ConvBlock1v1(nn.Module):
    def __init__(self, in_channel=3, pretrained_dict=None):
        super(ConvBlock1v1, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = self.bn2(out)
        return out


class ConvBlock2v1(nn.Module):
    def __init__(self, in_channel=16, out_channel=2, pretrained_dict=None):
        super(ConvBlock2v1, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.fc1 = nn.Linear(32*1*1, out_channel)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.avg_pool2d(out, 24)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class SPDBlock(nn.Module):
    def __init__(self):
        super(SPDBlock, self).__init__()

    def forward(self, left, right):
        if self.training:
            flag = random.randint(0, 2)
            if flag == 0:
                out = left
            elif flag == 1:
                out = right
            else:
                out = left + right
                # out = torch.cat((left, right), 0)
        else:
            out = left + right
            # out = torch.cat((left, right), 0)
        return out


class MyNetv2(nn.Module):
    def __init__(self):
        super(MyNetv2, self).__init__()
        self.rgb_conv_block = ConvBlock1v1(3)
        self.ir_conv_block = ConvBlock1v1(1)
        self.spd_block = SPDBlock()
        self.conv_block = ConvBlock2v1(16)

    def forward(self, x):
        # out = F.relu(self.conv1(x))
        rgb_in = x[:, :3, :, :].clone()
        ir_in = x[:, 3, :, :].unsqueeze(1).clone()
        rgb_out = self.rgb_conv_block(rgb_in)
        ir_out = self.ir_conv_block(ir_in)
        out = self.spd_block(rgb_out, ir_out)
        out = self.conv_block(out)

        return out


class MyNetv2CIFAR(nn.Module):
    def __init__(self, pretrained_dict=None):
        super(MyNetv2CIFAR, self).__init__()
        self.rgb_conv_block = ConvBlock1v1(3)
        self.ir_conv_block = ConvBlock1v1(1)
        self.spd_block = SPDBlock()
        self.conv_block = ConvBlock2v1(16, 2)

        if pretrained_dict:
            pretrained_dict = torch.load(pretrained_dict)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "fc1" not in k}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def forward(self, x):
        # out = F.relu(self.conv1(x))
        rgb_in = x[:, :3, :, :].clone()
        ir_in = x[:, 3, :, :].unsqueeze(1).clone()
        rgb_out = self.rgb_conv_block(rgb_in)
        ir_out = self.ir_conv_block(ir_in)
        out = self.spd_block(rgb_out, ir_out)
        out = self.conv_block(out)
        return out


if __name__ == "__main__":
    test_net = MyNetv2CIFAR(os.path.join(MODELS_DIR, "..", "pretrained", "MyNet_CIFAR10.pth"))
    pretrained_dict = torch.load(os.path.join(MODELS_DIR, "..", "pretrained", "MyNet_CIFAR10.pth"))
    test_dict = test_net.state_dict()
    for key in pretrained_dict.keys():
        if key in test_dict.keys():
            print(pretrained_dict[key])
            print(test_dict[key])
        else:
            print(key)