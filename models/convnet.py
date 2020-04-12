import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_net_001(nn.Module):
    def __init__(self, num_classes=10):
        super(conv_net_001, self).__init__()

        self.conv = nn.Conv2d(3, num_classes, 32)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        return out


class conv_net_002(nn.Module):
    def __init__(self, num_classes=10):
        super(conv_net_002, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 11),
            nn.ReLU(),
            nn.Conv2d(32, 64, 11),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 11)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.features(x)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        return out


class conv_net_003(nn.Module):
    def __init__(self, num_classes=10):
        super(conv_net_003, self).__init__()

        self.fea1 = nn.Sequential(
            nn.Conv2d(3, 32, 11),
            nn.ReLU(),
            nn.Conv2d(32, 64, 11),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 11)
        )

        self.fea2 = nn.Sequential(
            nn.Conv2d(3, 32, 7),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 5)
        )

        self.fea3 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 3)
        )

        # self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        # x1 = x
        # x2 = F.interpolate(x1, scale_factor=0.5, mode="bilinear")
        # x3 = F.interpolate(x2, scale_factor=0.5, mode="bilinear")

        out1 = self.fea1(x)
        out2 = self.fea2(x)
        out3 = self.fea3(x)

        out1 = self.pool(out1)
        out2 = self.pool(out2)
        out3 = self.pool(out3)

        out = torch.cat([out1, out2, out3], dim=2)
        out, _ = torch.max(out, dim=2, keepdim=True)
        out = out.view(out.size(0), -1)
        return out
