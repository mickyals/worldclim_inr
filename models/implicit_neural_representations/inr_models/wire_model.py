# define wire layer
# define wire model
# define logic for residual layer

import torch
import torch.nn as nn
import numpy
from helpers import get_logger
from utils.activations import wire_activation
from utils.Initializers import Initializer
from utils.positional_encodings import GaussianFourierFeatureTransform


LOGGER = get_logger(name = "WireModel", log_file="worldclim-dataset.log")

class WireLayer(nn.Module):
    def __init__(self, in_features, out_features, scale=10, omega=20, bias=True, weight_init=1.0, bias_init=2.0, dtype=torch.cfloat):
        super().__init__()

        LOGGER.debug("Initializing WireLayer")

        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.omega = omega
        self.bias = bias
        self.weight_init = weight_init
        self.bias_init = bias_init

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        Initializer.wire_init(self.linear, self.weight_init, self.bias_init, bias)

        self.activation = wire_activation(scale=self.scale, omega_w=self.omega)

    def forward(self, x):
        LOGGER.debug("forwarding WireLayer")
        out = self.linear(x)
        return self.activation(out)



class WireResidualLayer(nn.Module):
    def __init__(self, in_features, out_features, scale=10, omega=20, bias=True, weight_init=1.0, bias_init=2.0, dtype=torch.cfloat):
        super().__init__()

        LOGGER.debug("Initializing WireResidualLayer")
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.omega = omega
        self.bias = bias
        self.weight_init = weight_init
        self.bias_init = bias_init

        self.linear1 = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        self.linear2 = nn.Linear(out_features, out_features, bias=bias, dtype=dtype)

        Initializer.wire_init(self.linear1, self.weight_init, self.bias_init, bias)
        Initializer.wire_init(self.linear2, self.weight_init, self.bias_init, bias)

        self.activation = wire_activation(scale=self.scale, omega_w=self.omega)

    def forward(self, x):
        LOGGER.debug("forwarding WireResidualLayer")
        out = self.linear1(x)
        first_activation = self.activation(out)
        out = self.linear2(first_activation)
        out = out + x
        return self.activation(out)


class WireModel(nn.Module):
    def __init__(self, in_features, out_features, mapping_type='gauss', mapping_dim=4, mapping_scale=10, hidden_layers=5,hidden_features=256,
                 bias=True, final_bias=False, scale=2.0, omega=30.0, weight_init=1.0, bias_init=0.1, residual_net=False):
        super().__init__()
        LOGGER.debug("Initializing WireModel")

        self.net = []

        #positional encoding
        if mapping_type == 'no':
            LOGGER.debug("No positional encoding")
            self.net.append(WireLayer(in_features, hidden_features, scale, omega, bias, weight_init, bias_init, dtype=torch.float))
        else:
            LOGGER.debug("Using positional encoding")
            self.net.append(
            GaussianFourierFeatureTransform(type=mapping_type, input_dim=in_features, mapping_dim=mapping_dim, scale=mapping_scale))

            self.net.append(WireLayer(mapping_dim, hidden_features, scale, omega, bias, weight_init, bias_init, dtype=torch.float))

        # build hidden layers
        for i in range(hidden_layers):
            LOGGER.debug("Adding hidden layer")
            if residual_net:
                LOGGER.debug("Adding residual layer")
                self.net.append(WireResidualLayer(hidden_features, hidden_features, scale, omega, bias, weight_init, bias_init))
            else:
                LOGGER.debug("Adding normal layer")
                self.net.append(WireLayer(hidden_features, hidden_features, scale, omega, bias, weight_init, bias_init))

        # Define and initialize the final linear layer
        final_layer = nn.Linear(hidden_features, out_features, bias=final_bias, dtype=torch.cfloat)
        with torch.no_grad():
            nn.init.uniform_(final_layer.weight, -weight_init, weight_init)
        self.net.append(final_layer)

        self.net = nn.Sequential(*self.net)



    def forward(self, x):
        LOGGER.debug("FORWARD WireModel")
        return self.net(x)#.real