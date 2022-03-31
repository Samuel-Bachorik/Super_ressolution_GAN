import torchvision.models as models
import torch.nn.functional as F
from torch import nn
import torch
class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

        vgg19 = models.vgg19(pretrained=True).eval()

        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:36])

        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

        # Normalization
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr, hr):
        sr = sr.sub(self.mean).div(self.std)
        hr = hr.sub(self.mean).div(self.std)

        loss = F.l1_loss(self.feature_extractor(sr), self.feature_extractor(hr))

        return loss