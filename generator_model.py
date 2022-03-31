import torch
import torch.nn as nn

class ResidualBlock(torch.nn.Module):
    def __init__(self, filters, kernel_size=3, init_gain=0.1):
        super(ResidualBlock, self).__init__()

        self.conv0  = nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn0   = nn.BatchNorm2d(filters)
        self.act0  = nn.PReLU(num_parameters=64)

        self.conv1 = nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn1   = nn.BatchNorm2d(filters)
        self.act1  = nn.PReLU(num_parameters=64)

    def forward(self, x):
        y = self.act0(self.bn0(self.conv0(x)))
        y = self.bn1(self.conv1(y))

        return y + x


class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c * scale_factor ** 2, 3, 1, 1)
        self.ps   = nn.PixelShuffle(scale_factor)
        self.act  = nn.PReLU(num_parameters=in_c)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class Super_ress_model(torch.nn.Module):
    def __init__(self):
        super(Super_ress_model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input  = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=9 // 2)
        self.bn     = nn.Identity()
        self.act    = nn.PReLU(num_parameters=64)

        self.blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(18)])

        self.conv1  = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=3 // 2)
        self.bn1    = nn.BatchNorm2d(64)

        self.up     = nn.Sequential(UpsampleBlock(64,2),UpsampleBlock(64,2))
        self.output = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=9 // 2)

    def forward(self, x):

        input = self.act(self.bn(self.input(x)))
        x = self.blocks(input)
        x = self.bn1(self.conv1(x)) + input
        x = self.up(x)
        x = self.output(x)

        return torch.sigmoid(x)