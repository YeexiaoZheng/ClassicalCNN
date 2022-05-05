import torch.nn as nn
import torch

# GoogLeNet Inception模块
class Inception(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2_1, out_channels_2_2, out_channels_3_1, out_channels_3_2, out_channels_4):
        super(Inception, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels_1, kernel_size=1),
            nn.BatchNorm2d(out_channels_1),
            nn.ReLU()
        ) 

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels_2_1, kernel_size=1),
            nn.BatchNorm2d(out_channels_2_1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels_2_1, out_channels=out_channels_2_2,kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels_2_2),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels_3_1, kernel_size=1),
            nn.BatchNorm2d(out_channels_3_1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels_3_1, out_channels=out_channels_3_2, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels_3_2),
            nn.ReLU()
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels_4, kernel_size=1),
            nn.BatchNorm2d(out_channels_4),
            nn.ReLU()
        )

    def forward(self, x):
        x_1 = self.branch1(x)
        x_2 = self.branch2(x)
        x_3 = self.branch3(x)
        x_4 = self.branch4(x)
        x = torch.cat([x_1, x_2, x_3, x_4], 1)
        return x

# 辅助分类器
class AuxClassifier(nn.Module):
    def __init__(self, in_channels, label_num=10, dropout=0.5):
        super(AuxClassifier, self).__init__()

        self.average_pool = nn.AvgPool2d(kernel_size=5, stride=3) 

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, label_num),
            nn.ReLU()
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.average_pool(x)
        x = self.conv(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

# GoogLeNet
class GoogLeNet(nn.Module):
    def __init__(self, label_num=10, dropout=0.5, aux=False):
        super(GoogLeNet, self).__init__()

        self.aux = aux

        self.conv_pool = nn.Sequential(
            # (1*28*28) -> (8*28*28)
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # (8*28*28) -> (8*14*14)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # (8*14*14) -> (8*14*14)
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            # (8*14*14) -> (24*14*14)
            nn.Conv2d(in_channels=8, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            # (24*14*14) -> (24*7*7)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.inceptions_1 = nn.Sequential(
            Inception(24, 8, 12, 16, 2, 4, 4),          # inception3a
            Inception(32, 16, 16, 24, 4, 12, 8),        # inception3b
            # (24*7*7) -> (24*3*3)
            nn.MaxPool2d(kernel_size=3, stride=2),      # MaxPool 3*3+2(S)
            Inception(60, 24, 12, 26, 2, 6, 8)          # inception4a
        )

        self.aux1 = AuxClassifier(64, label_num, dropout)
        
        self.inceptions_2 = nn.Sequential(
            Inception(64, 20, 14, 28, 3, 8, 8),         # inception4b
            Inception(64, 16, 16, 32, 3, 8, 8),         # inception4c
            Inception(64, 14, 18, 36, 4, 8, 8),         # inception4d
            Inception(66, 32, 20, 40, 4, 16, 16),       # inception4e
            # (24*3*3) -> (24*1*1)
            nn.MaxPool2d(kernel_size=3, stride=2)       # MaxPool 3*3+2(S)
        )

        self.aux2 = AuxClassifier(66, label_num, dropout)

        self.inceptions_3 = nn.Sequential(
            Inception(104, 32, 20, 40, 4, 16, 16),      # inception5a
            Inception(104, 48, 24, 48, 6, 16, 16)       # inception5b
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, label_num),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_pool(x)
        
        x = self.inceptions_1(x)
        if self.training and self.aux:
            x_aux_1 = self.aux1(x)

        x = self.inceptions_2(x)
        if self.training and self.aux:
            x_aux_2 = self.aux2(x)

        x = self.inceptions_3(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        
        if self.aux:
            return x, x_aux_1, x_aux_2

        return x