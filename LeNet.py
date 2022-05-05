import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, label_num=10):
        super(LeNet, self).__init__()

        self.conv_pool_1 = nn.Sequential(
            # 卷积层 (1*28*28) -> 6*28*28)
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),                                                          
            # 池化层 (6*28*28) -> (6*14*14)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_pool_2 = nn.Sequential(
            # 卷积层 (6*14*14) -> (16*10*10)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            # 池化层 (16*10*10) -> (16*5*5)
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            # 将卷积池化后的tensor拉成向量
            nn.Flatten(),
            # 全连接层 16*5*5 -> 120
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            # 全连接层 120 -> 84
            nn.Linear(120, 84),
            nn.ReLU(),
            # 全连接层 84 -> 10
            nn.Linear(84, label_num)
        )

    def forward(self, x):
        x = self.conv_pool_1(x)
        x = self.conv_pool_2(x)
        x = self.fc(x)
        return x