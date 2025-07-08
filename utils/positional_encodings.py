import torch
import torch.nn as nn
from helpers import get_logger

LOGGER = get_logger(name = "PositionalEncodings", log_file="worldclim-dataset.log")


########################################################################################################################
### GAUSSIAN FOURIER POSITIONAL ENCODING
#######################################################################################################################

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




#######################################################################################################################
### DOUBLE FOURIER SPHERICAL POSITIONAL ENCODING
#######################################################################################################################

class SphericalFourierFeatureTransform(nn.Module):
    def __init__(self,  omegas: list[float], scale: list[int] ):
        """
        Initialize the spherical positional encoding.

        Args:
            omegas (list[float]): The frequency scaling factors for latitude and longitude.
            scale (list[int]): The number of scales for latitude and longitude.
        """
        super().__init__()

        # coords = [lat, lon]
        # omegas = [omega_lat, omega_lon]
        # scale = [scale_lat, scale_lon]

        # general function for positional encoding is sin(omega_lat^(scale_lat_i/(len(scale_lat)) - 1) * lat)
        LOGGER.debug("Initializing SphericalFourierFeatureTransform")
        self.omega_lat, self.omega_lon = omegas
        self.scale_lat, self.scale_lon = scale

        self.freq_lat = self._compute_freq(self.omega_lat, self.scale_lat)
        self.freq_lon = self._compute_freq(self.omega_lon, self.scale_lon)


    def _compute_freq(self, omega, scale):
        """
        Compute the frequency of the spherical positional encoding as a function of the scale.

        Args:
            omega (float): The frequency scaling factor.
            scale (int): The number of scales.

        Returns:
            A tensor of shape [scale] containing the frequency of the spherical positional encoding.
        """
        LOGGER.debug("Computing the frequency of the spherical positional encoding.")
        if scale == 1:
            return torch.ones(1)
        s = torch.arange(scale)
        return omega ** (s / (scale - 1))


    def forward(self, coords):
        """
        Compute the spherical positional encoding for a batch of coordinates.

        Args:
            coords: A tensor of shape [B, 2] containing the latitude and longitude coordinates.

        Returns:
            A tensor of shape [B, total_dim] containing the spherical positional encoding.
        """
        # coords = [lat, lon]

        LOGGER.debug("Computing the spherical positional encoding.")
        B = coords.shape[0]
        lat = coords[:, 0] * (torch.pi / 180)  # shape [B]
        lon = coords[:, 1] * (torch.pi / 180)  # shape [B]

        # scaled lat and lon
        lat_scaled = 2 * torch.pi * lat.unsqueeze(1) * self.freq_lat.unsqueeze(0)  # shape [B, scale_lat]
        lon_scaled = 2 * torch.pi * lon.unsqueeze(1) * self.freq_lon.unsqueeze(0)  # shape [B, scale_lon]

        # sin and cos of scaled lat and lon
        sin_lat = torch.sin(lat_scaled)  # shape [B, scale_lat]
        cos_lat = torch.cos(lat_scaled)  # shape [B, scale_lat]
        sin_lon = torch.sin(lon_scaled)  # shape [B, scale_lon]
        cos_lon = torch.cos(lon_scaled)  # shape [B, scale_lon]

        # basic positions
        lat_terms = torch.cat([sin_lat, cos_lat], dim=-1)  # shape [B, 2 * scale_lat]
        lon_terms = torch.cat([sin_lon, cos_lon], dim=-1)  # shape [B, 2 * scale_lon]

        # interaction terms
        sin_lat_expanded = sin_lat.unsqueeze(-1)  # shape [B, scale_lat, 1]
        cos_lat_expanded = cos_lat.unsqueeze(-1)  # shape [B, scale_lon, 1]
        sin_lon_expanded = sin_lon.unsqueeze(-2)  # shape [B, 1, scale_lon]
        cos_lon_expanded = cos_lon.unsqueeze(-2)  # shape [B, 1, scale_lon]

        term1 = (cos_lat_expanded * cos_lon_expanded)  # shape [B, scale_lat, scale_lon]
        term2 = (cos_lat_expanded * sin_lon_expanded)  # shape [B, scale_lat, scale_lon]
        term3 = (sin_lat_expanded * cos_lon_expanded)  # shape [B, scale_lat, scale_lon]
        term4 = (sin_lat_expanded * sin_lon_expanded)  # shape [B, scale_lat, scale_lon]

        interaction = torch.cat([term1, term2, term3, term4], dim=1)  # shape [B, 4, scale_lat, scale_lon]
        interaction = interaction.view(B, -1)  # shape [B, 4 * scale_lat * scale_lon]

        # concatenate all terms
        output = torch.cat([lat_terms, lon_terms, interaction], dim=-1)  # [B, total_dim]
        return output



##########################################################################################################################
## SPHERE2VEC POSITIONAL ENCODING
##########################################################################################################################




class Sphere2Vec(nn.Module):
    def __init__(self, omega: int, scale: int, mode: str):
        super().__init__()

        self.omega = omega
        self.scale = scale
        self.mode = mode

        # Precompute frequencies based on omega and scale
        self.freqs = self._compute_freqs()

    def _compute_freqs(self):
        """
        Compute the frequency for each scale step (this will return an array of size [scale]).
        """
        s = torch.arange(self.scale)
        return self.omega ** (s / (self.scale - 1))  # Returns frequencies for each scale step

    def forward(self, coords):
        """
        Compute positional encoding for spherical coordinates (latitude, longitude).

        coords: A tensor of shape [B, 2] where:
            coords[:, 0] = latitudes in degrees (ranging from -90 to 90)
            coords[:, 1] = longitudes in degrees (ranging from -180 to 180)
        """
        B = coords.shape[0]
        lat = coords[:, 0] * torch.pi / 180  # Convert latitudes to radians
        lon = coords[:, 1] * torch.pi / 180  # Convert longitudes to radians

        # Scale frequencies
        freq = self.freqs.to(coords.device)  # [scale]
        lat_scaled = lat.unsqueeze(1) * freq.unsqueeze(0)  # [B, scale]
        lon_scaled = lon.unsqueeze(1) * freq.unsqueeze(0)  # [B, scale]

        # Calculate sin and cos of scaled latitudes and longitudes
        sin_lat = torch.sin(lat_scaled)  # [B, scale]
        cos_lat = torch.cos(lat_scaled)  # [B, scale]
        sin_lon = torch.sin(lon_scaled)  # [B, scale]
        cos_lon = torch.cos(lon_scaled)  # [B, scale]

        if self.mode == "SphereC":
            # SphereC: Encode interaction terms for spherical coordinates
            terms = torch.cat([
                sin_lat,  # [B, scale]
                cos_lat * cos_lon,  # [B, scale]
                cos_lat * sin_lon  # [B, scale]
            ], dim=-1)  # [B, 3 * scale]

        elif self.mode == "SphereM":
            # SphereM: Extended interaction terms for spherical coordinates
            base_cos_lat = torch.cos(lat).unsqueeze(1)  # [B, 1]
            base_sin_lon = torch.sin(lon).unsqueeze(1)  # [B, 1]
            base_cos_lon = torch.cos(lon).unsqueeze(1)  # [B, 1]

            terms = torch.cat([
                sin_lat,  # [B, scale]
                cos_lat * base_cos_lon,  # [B, scale]
                cos_lat * cos_lon,  # [B, scale]
                cos_lat * base_sin_lon,  # [B, scale]
                cos_lat * sin_lon  # [B, scale]
            ], dim=-1)  # [B, 5 * scale]

        return terms  # Output shape: [B, output_dim]






