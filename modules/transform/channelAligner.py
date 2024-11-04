from modules.layers.conv import conv, conv1x1, conv3x3, deconv
from torch import nn


class Channel_aligner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            conv3x3(64, 256),
            nn.LeakyReLU(inplace=True),
            conv3x3(256, 256),
            nn.LeakyReLU(inplace=True),
            conv3x3(256, 256),
            nn.LeakyReLU(inplace=True),
            conv3x3(256, 256),
            nn.LeakyReLU(inplace=True),
        )

        self.conv2 = conv3x3(256, 64)
        self.conv3 = conv3x3(256, 64)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)  # adaptive的参数是输出的size，而普通的参数则是过程的kernel_size
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)

    # feature2 is guided
    def forward(self, feature1, feature2):
        identity = feature2

        # 计算beta
        out1 = self.conv1(feature1)
        out1 = self.conv2(out1)
        beta = self.avgpool1(out1)

        # 计算gamma
        out2 = self.conv1(feature2)
        out2 = self.conv3(out2)
        gamma = self.avgpool2(out2)

        # 池化==》广播
        out = gamma * identity + beta  # 这里是否直接通过乘法和加法来实现？ # 这里通过广播来实现？
        # print("beta,gamma:")
        # print(beta,gamma)
        return out, beta, gamma  # 计算bpp的时候需要，但是优化的时候不需要，因为损失加上一个常数不重要
