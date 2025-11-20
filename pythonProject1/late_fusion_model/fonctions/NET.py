import torch.nn as nn
import torch.nn.functional as F
import torch
class Net(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 100, 3, padding=1)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(100, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 80, 3, padding=1)
        self.bn7   = nn.BatchNorm2d(80)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(80, 80)
        self.bn_fc1 = nn.LayerNorm(80)
        self.out = nn.Linear(80, num_labels)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool2(F.relu(self.bn7(self.conv7(x))))
        x = self.gap(x)                     # (B,80,1,1)
        x = torch.flatten(x, 1)             # (B,80)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        return self.out(x)