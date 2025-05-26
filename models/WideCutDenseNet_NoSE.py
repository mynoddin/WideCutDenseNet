
# models/WideCutDenseNet_NoSE.py
# Variant of WideCutDenseNet with SE-Attention removed

import torch.nn as nn
import torch.nn.functional as F

class WideCutDenseNet_NoSE(nn.Module):
    def __init__(self, num_classes=10):
        super(WideCutDenseNet_NoSE, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.block1 = self._make_layer(64, 128)
        self.block2 = self._make_layer(128, 256)
        self.classifier = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.classifier(out)
