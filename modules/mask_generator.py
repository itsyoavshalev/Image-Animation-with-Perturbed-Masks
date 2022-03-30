from torch import nn
import torch
import torch.nn.functional as F
from modules.util import HourglassNoRes, Hourglass, AntiAliasInterpolation2d

class MaskGenerator(nn.Module):
    def __init__(self, block_expansion, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0):
        super(MaskGenerator, self).__init__()

        self.predictor = HourglassNoRes(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.ref = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(7, 7),
                            padding=pad)

        self.sigmoid = nn.Sigmoid()
        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        mask = self.ref(feature_map) / self.temperature
        mask = self.sigmoid(mask)

        return mask
