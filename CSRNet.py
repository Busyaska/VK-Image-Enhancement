import torch
import torch.nn as nn


INPUT_SIZE = 256 


class ConditionBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.downsample(x)
        return x


class ConditionNet(nn.Module):

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.block1 = ConditionBlock(in_channels, 16)
        self.block2 = ConditionBlock(16, 32)
        self.block3 = ConditionBlock(32, 64)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        return x.flatten(1)


class EnhanceNet(nn.Module):

    GAIN_MIN = 0.1
    GAIN_MAX = 3.0

    def __init__(self):
        super().__init__()
        self.condition_net = ConditionNet(in_channels=3)

        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(32, 3)
        )

        self._initialize_weights()

    def forward(self, x):
        condition_vector = self.condition_net(x)
        raw = self.head(condition_vector)

        gains = self.GAIN_MIN + torch.sigmoid(raw) * (self.GAIN_MAX - self.GAIN_MIN)
        return gains

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
