import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import os
from tqdm import tqdm
from helpers import get_logger, set_seed, set_device, instantiate_from_config

LOGGER = get_logger(name = "NeuralDMD", log_file="worldclim-dataset.log")



class NeuralDMD(L.LightningModule):
    def __init__(self, config):
        super(NeuralDMD, self).__init__()
        self.config = config

        # define embeddings
        self.spatial_position_embedding = instantiate_from_config(config['SpatialPositionEmbedding'])
        self.time_position_embedding = instantiate_from_config(config['TimePositionEmbedding'])

        # define networks
        self.spatial_network = instantiate_from_config(config['SpatialNetwork'])
        self.spectral_network = instantiate_from_config(config['SpectralNetwork'])
        self.initial_condition_network = instantiate_from_config(config['InitialConditionNetwork'])

        # define loss
        loss_config = self.config.get('Loss', {})
        loss_type = loss_config.get('type', 'MSELoss')
        loss_params = loss_config.get('params', {})
        loss_class = getattr(nn, loss_type, nn.MSELoss)
        self.loss = loss_class(**loss_params)

        self.save_hyperparameters()


    def compute_modes(self,spatial_coords, past_time_coords):
        space_coords = self.spatial_position_embedding(spatial_coords)
        # (num_embeddings) -> (R)
        spatial_modes = self.spatial_network(space_coords)

        # (C) -> (num_embeddings_time)
        past_time_embedding = self.time_position_embedding(past_time_coords)
        # (num_embeddings_time) -> (B, R)
        spectral_modes = self.spectral_network(past_time_embedding)

        return spatial_modes, spectral_modes

    def compute_dynamics(self,spatial_modes, spectral_modes, time_coords, initial_conditions):
        time_coords = self.time_position_embedding(time_coords)
        # (B, n) -> (B, R)
        dynamics = spatial_modes * torch.exp(spectral_modes * time_coords)  * initial_conditions

        return dynamics

    def forward(self, spatial_coords, past_time_coords, time_coords, initial_conditions):

        spatial_modes, spectral_modes = self.compute_modes(spatial_coords, past_time_coords)
        initial_conditions = self.initial_condition_network(initial_conditions)
        dynamics = self.compute_dynamics(spatial_modes, spectral_modes, time_coords, initial_conditions)

        return spatial_modes, spectral_modes, initial_conditions, dynamics


    def training_step(self, batch, batch_idx):
        pass
