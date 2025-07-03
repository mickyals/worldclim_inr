import torch
import torch.nn as nn
from helpers import get_logger

LOGGER = get_logger(name = "PositionalEncodings", log_file="worldclim-dataset.log")


class GaussianFourierFeatureTransform(nn.Module):
    def __init__(self, type, input_dim, mapping_dim, scale):
        super().__init__()

        self.type = type
        self.input_dim = input_dim
        self.mapping_dim = self.get_mapping_dim(mapping_dim)
        self.scale = scale
        self.preprocessing_param = self.get_preprocessing_param() * self.scale

    def get_preprocessing_param(self):
        assert self.type in ["no", "basic", "gauss"], "must define pre-processing"

        if self.type.find("gauss") >= 0:
            param = torch.randn(self.input_dim, self.mapping_dim // 2)
        elif self.type == "basic":
            param = torch.eye(self.input_dim)
        else:
            param = -1

        return param

    def forward(self, x):
        x_transform = 2 * torch.pi * torch.matmul(x, self.preprocessing_param) if self.type != "no" else None
        return torch.cat([torch.sin(x_transform), torch.cos(x_transform)], dim=-1) if self.type != "no" else x

    def get_mapping_dim(self, mapping_dim):
        if self.type == "no":
            mapping_dim = self.input_dim
        elif self.type == "basic":
            mapping_dim = self.input_dim * 2
        elif self.type.find("gauss") >= 0:
            mapping_dim = mapping_dim

        return mapping_dim