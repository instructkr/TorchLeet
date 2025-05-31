import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
