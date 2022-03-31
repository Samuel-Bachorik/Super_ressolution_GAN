import torch
import torch.nn as nn

class Sobel_operator:
    def __init__(self):

        self.conv = nn.Conv2d(in_channels = 3, out_channels = 2, kernel_size = 3)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv.to(self.device)
        torch.nn.init.zeros_(self.conv.bias)

        Gx = torch.nn.Parameter(torch.tensor([[1.,0.,-1.],[2.,0.,-2.],[1.,0.,-1.]]))
        Gy = torch.nn.Parameter(torch.tensor([[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]]))

        with torch.no_grad():
            for x in range(3):
                self.conv.weight[0][x][:] = Gx
                self.conv.weight[1][x][:] = Gy

    def find_edges(self,images):

        edges = self.conv(images)
        return edges