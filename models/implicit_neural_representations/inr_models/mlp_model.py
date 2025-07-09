# define mlp layer
# define mlp model
# define logic for residual layer

import torch
import torch.nn as nn
from helpers import get_logger
from utils.Initializers import Initializer
from utils.positional_encodings import ENCODER_REGISTRY

LOGGER = get_logger(name = "MLPModel", log_file="worldclim-dataset.log")

class MLPLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_init=0.1, bias_init=None):
        super().__init__()

        LOGGER.info("MLP LAYER")
        self.in_features = in_features
        self.out_features = out_features
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.activation = nn.ReLU()

        # Define the linear layer using nn.Linear
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Initialize the weights of the linear layer
        Initializer.mlp_init(self.linear, self.weight_init, self.bias_init, bias)

    def forward(self, x):
        LOGGER.debug("FORWARD PASS")
        out = self.linear(x)
        return self.activation(out)

class MLPResidualLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_init=0.1, bias_init=None):
        super().__init__()
        LOGGER.info("MLP RESIDUAL LAYER")
        self.in_features = in_features
        self.out_features = out_features
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.activation = nn.ReLU()

        # Define the linear layer using nn.Linear
        self.first_linear = nn.Linear(in_features, out_features, bias=bias)
        self.last_linear = nn.Linear(in_features, out_features, bias=bias)

        # Initialize the weights of the linear layer
        Initializer.mlp_init(self.first_linear, self.weight_init, self.bias_init, bias)
        Initializer.mlp_init(self.last_linear, self.weight_init, self.bias_init, bias)

    def forward(self, x):
        LOGGER.debug("FORWARD PASS")
        out = self.first_linear(x)
        first_activation = self.activation(out)
        out = self.last_linear(first_activation)
        out = out + x
        return self.activation(out)


class MLPModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=128,
                 hidden_layers = 5, bias=True, weight_init=0.1, bias_init=None,
                 residual_net=False, encoding=None, **encoder_kwargs):
        super().__init__()
        LOGGER.info("Initializing MLP MODEL")
        self.net = []

        # positional encoder
        if encoding is None:
            self.encoder = nn.Identity()
            encoded_dim = in_features
        else:
            encoder_cls = ENCODER_REGISTRY[encoding]
            self.encoder = encoder_cls(**encoder_kwargs)
            with torch.no_grad():
                dummy_input = torch.zeros(1, in_features)
                encoded_dim = self.encoder(dummy_input).shape[-1]

        # Add the first MLP layer
        self.net.append(MLPLayer(encoded_dim, hidden_features, bias=bias, weight_init=weight_init, bias_init=bias_init))

        # Add the residual layers
        for i in range(hidden_layers):
            if residual_net:
                self.net.append(MLPResidualLayer(hidden_features, hidden_features, bias=bias, weight_init=weight_init, bias_init=bias_init))
            else:
                self.net.append(MLPLayer(hidden_features, hidden_features, bias=bias, weight_init=weight_init, bias_init=bias_init))
        # Define and initialize the final linear layer
        final_layer = nn.Linear(hidden_features, out_features, bias=bias)
        with torch.no_grad():
            nn.init.uniform_(final_layer.weight, -weight_init, weight_init)
        self.net.append(final_layer)

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        LOGGER.debug("FORWARDING MLP MODEL")
        x = self.encoder(x)
        return self.net(x)
