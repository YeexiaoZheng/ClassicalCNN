import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    multiplier = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.multiplier, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * self.multiplier),
        )

        self.shortcut = nn.Sequential()
        if in_channels != out_channels * self.multiplier or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels * self.multiplier, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels * self.multiplier)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.conv2(self.conv1(x))
        shortcut = self.shortcut(x)
        return self.relu(residual + shortcut)

class Bottleneck(nn.Module):
    multiplier = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.multiplier, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels * self.multiplier)
        )

        self.shortcut = nn.Sequential()
        if in_channels != out_channels * self.multiplier or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels * self.multiplier, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels * self.multiplier)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        resiudual = self.conv3(self.conv2(self.conv1(x)))
        shortcut = self.shortcut(x)
        return self.relu(resiudual + shortcut)

class ResNet(nn.Module):

    def __init__(self, layer_num=18, label_num=10):
        super(ResNet, self).__init__()
        self.base_channels = 64

        block_type, block_nums = self.res_net_params(layer_num)

        self.conv_pool_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.base_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        self.res_layers = nn.Sequential(
            self.res_layer(block_type, 64, block_nums[0], stride=1),
            self.res_layer(block_type, 128, block_nums[1], stride=2),
            self.res_layer(block_type, 256, block_nums[2], stride=2),
            self.res_layer(block_type, 512, block_nums[3], stride=2)
        )

        # 平均池化，平均池化成1*1
        self.avg_pool_layer = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * block_type.multiplier, label_num)
        )

    def res_layer(self, block_type, out_channel, block_num, stride):
        blocks = []
        for _ in range(block_num):
            new_block = block_type(in_channels=self.base_channels, out_channels=out_channel, stride=stride)
            blocks.append(new_block)
            self.base_channels = out_channel * new_block.multiplier
        return nn.Sequential(*blocks)
    
    def res_net_params(self, layer_num):
        if layer_num == 18:
            return BasicBlock, [2, 2, 2, 2]
        if layer_num == 34:
            return BasicBlock, [3, 4, 6, 3]
        if layer_num == 50:
            return Bottleneck, [3, 4, 6, 3]   
        if layer_num == 101:
            return Bottleneck, [3, 4, 23, 3]
        if layer_num == 152:
            return Bottleneck, [3, 8, 36, 3]
        
    def forward(self, x):
        x = self.conv_pool_layer(x)
        x = self.res_layers(x)
        x = self.avg_pool_layer(x)
        x = self.fc_layer(x)
        return x