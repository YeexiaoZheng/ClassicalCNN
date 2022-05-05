import torch.nn as nn

# 由于论文中的AlexNet输入图像维度过大，导致直接套用论文的参数无法训练，因此模型参数为自己填写
class AlexNet(nn.Module):
    def __init__(self, label_num=10, dropout=0):
        super(AlexNet,self).__init__()

        self.conv_pool_1 = nn.Sequential(
            # 卷积层 (1*28*28) -> (24*28*28)
            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 池化层 (24*28*28) -> (24*14*14)
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LocalResponseNorm(size=3)
        )

        self.conv_pool_2 = nn.Sequential(
            # 卷积层 (24*14*14) -> (64*14*14)
            nn.Conv2d(in_channels=24, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 池化层 (64*14*14) -> (64*7*7)
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LocalResponseNorm(size=3)
        )

        self.conv_pool_3 = nn.Sequential(
            # 卷积层 (64*7*7) -> (96*7*7)
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 卷积层 (96*7*7) -> (96*7*7)
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 卷积层 (96*7*7) -> (64*7*7)
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 池化层 (64*7*7) -> (64*3*3)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            # 将卷积池化后的tensor拉成向量
            nn.Flatten(),
            # dropout
            nn.Dropout(dropout),
            # 全连接层 (64*3*3) -> (512)
            nn.Linear(64 * 3 * 3, 512),
            nn.ReLU(),
            # dropout
            nn.Dropout(dropout),
            # 全连接层 (512) -> (512)
            nn.Linear(512, 512),
            nn.ReLU(),
            # dropout
            nn.Dropout(dropout),
            # 全连接层 (512) -> (10)
            nn.Linear(512, label_num)
        )

    def forward(self,x):
        x = self.conv_pool_1(x)
        x = self.conv_pool_2(x)
        x = self.conv_pool_3(x)
        x = self.fc(x)
        return x