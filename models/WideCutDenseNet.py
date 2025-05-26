
import torch
import torch.nn as nn
import torch.nn.functional as F

class CutoutDropout(nn.Module):
    def __init__(self, mask_size=8, p=0.5):
        super(CutoutDropout, self).__init__()
        self.mask_size = mask_size
        self.p = p

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p:
            return x
        _, _, h, w = x.size()
        mask = torch.ones_like(x)
        y = torch.randint(0, h, (1,))
        x_pos = torch.randint(0, w, (1,))
        y1 = torch.clamp(y - self.mask_size // 2, 0, h)
        y2 = torch.clamp(y + self.mask_size // 2, 0, h)
        x1 = torch.clamp(x_pos - self.mask_size // 2, 0, w)
        x2 = torch.clamp(x_pos + self.mask_size // 2, 0, w)
        mask[:, :, y1:y2, x1:x2] = 0
        return x * mask

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_rate, in_channels, kernel_size=1)
        self.se = SEBlock(in_channels)
        self.cutout = CutoutDropout(mask_size=8, p=0.25)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(torch.cat([x, out1], 1))))
        out = torch.cat([x, out1, out2], 1)
        out = self.conv3(out)
        out = self.se(out)
        out = self.cutout(out)
        return x + out

class WideCutDenseNet(nn.Module):
    def __init__(self, num_classes=10, depth=28, widen_factor=10, growth_rate=16):
        super(WideCutDenseNet, self).__init__()
        self.init_conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        channels = 16
        layers = []
        for stage in range(3):
            for _ in range((depth - 4) // 6):
                layers.append(ResidualDenseBlock(channels, growth_rate))
            layers.append(nn.Conv2d(channels, channels * 2, kernel_size=1))
            layers.append(nn.BatchNorm2d(channels * 2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            channels *= 2
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, num_classes)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
