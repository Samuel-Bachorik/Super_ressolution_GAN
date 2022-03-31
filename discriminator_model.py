import torch
import torch.nn as nn

class Discriminator_block(torch.nn.Module):
    def __init__(self, in_filters, out_filters, first_block= False):
        super(Discriminator_block, self).__init__()
        self.first_block = first_block

        self.conv0  = nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(out_filters)
        self.act0   = nn.LeakyReLU(0.2, inplace=True)


        self.conv1  = nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1)
        self.bn1    = nn.BatchNorm2d(out_filters)
        self.act1   = nn.LeakyReLU(0.2, inplace=True)

        torch.nn.init.xavier_uniform_(self.conv0.weight)
        torch.nn.init.zeros_(self.conv0.bias)

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)



    def forward(self,x):

        x = self.conv0(x)
        if not self.first_block:
            x = self.bn0(x)

        x = self.act0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hr_shape = (192, 192)


        self.input_shape = (3, *hr_shape)
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)


        self.block0 = Discriminator_block(3, 64,True)
        self.block1 = Discriminator_block(64, 128, False)
        self.block2 = Discriminator_block(128, 256, False)
        self.block3 = Discriminator_block(256, 512, False)

        self.out    = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)

        torch.nn.init.xavier_uniform_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)


    def forward(self,x):

        x = self.block0.forward(x)
        x = self.block1.forward(x)
        x = self.block2.forward(x)
        x = self.block3.forward(x)

        x = self.out(x)

        return x
