import torch
import torch.nn as nn
import numpy
import math
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

class SphericalFourierFeatureTransform(nn.Module):
    def __init__(self,  omegas: list[float], scale: list[int] ):
        super().__init__()

        # coords = [lat, lon]
        # omegas = [omega_lat, omega_lon]
        # scale = [scale_lat, scale_lon]

        # general function for positional encoding is sin(omega_lat^(scale_lat_i/(len(scale_lat)) - 1) * lat)

        self.omega_lat, self.omega_lon = omegas
        self.scale_lat, self.scale_lon = scale

        self.freq_lat = self._compute_freq(self.omega_lat, self.scale_lat)
        self.freq_lon = self._compute_freq(self.omega_lon, self.scale_lon)

    def _compute_freq(self, omega, scale):
        if scale == 1:
            return torch.ones(1)
        s = torch.arange(scale)
        return omega ** (s / (scale - 1))


    def forward(self, coords):

        # coords = [lat, lon]
        B = coords.shape[0]
        lat = coords[:, 0] * (torch.pi / 180) # shape [B]
        lon = coords[:, 1] * (torch.pi / 180) # shape [B]

        # scaled lat and lon
        lat_scaled = 2 * torch.pi * lat.unsqueeze(1) * self.freq_lat.unsqueeze(0) # shape [B, scale_lat]
        lon_scaled = 2 * torch.pi * lon.unsqueeze(1) * self.freq_lon.unsqueeze(0) # shape [B, scale_lon]

        # sin and cos of scaled lat and lon
        sin_lat = torch.sin(lat_scaled) # shape [B, scale_lat]
        cos_lat = torch.cos(lat_scaled) # shape [B, scale_lat]
        sin_lon = torch.sin(lon_scaled) # shape [B, scale_lon]
        cos_lon = torch.cos(lon_scaled) # shape [B, scale_lon]

        # basic positions
        lat_terms = torch.cat([sin_lat, cos_lat], dim=-1) # shape [B, 2 * scale_lat]
        lon_terms = torch.cat([sin_lon, cos_lon], dim=-1) # shape [B, 2 * scale_lon]

        # interaction terms
        lat_expanded = lat_terms.unsqueeze(-1) # shape [B, 2 * scale_lat, 1]
        lon_expanded = lon_terms.unsqueeze(1) # shape [B, 1, 2 * scale_lon]
        interaction = lat_expanded * lon_expanded # shape [B, 2 * scale_lat, 2 * scale_lon]

        interaction = interaction.view(B, -1) # shape [B, 4 * scale_lat * scale_lon]


        output = torch.cat([lat_terms, lon_terms, interaction], dim=-1)  # [B, total_dim]
        return output










