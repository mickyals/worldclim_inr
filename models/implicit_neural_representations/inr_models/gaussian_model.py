# define gaussian layer
# define gaussian model
# define logic for residual layer

import torch
import torch.nn as nn
import numpy
import torch.nn.functional as F
from helpers import get_logger
from models.implicit_neural_representations.activations import gaussian_activation

LOGGER = get_logger(name = "GaussianModel", log_file="worldclim-dataset.log")


class GaussianLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=30.0, weight_init=0.1):
        """
        Initializes a GaussianLayer.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): Whether to include bias terms in the linear layer. Defaults to True.
            scale (float, optional): The scaling factor for the Gaussian activation function. Defaults to 30.0.
            weight_init (float, optional): The initialization value for the weights of the linear layer. Defaults to 0.1.
        """

        super().__init__()
        LOGGER.info("GAUSSIAN LAYER")
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.weight_init = weight_init
        self.activation = gaussian_activation(self.scale)

        # Define the linear layer using nn.Linear
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Initialize the weights of the linear layer
        self._init_weights()

    def _init_weights(self):
        """
        Initializes the weights of the linear layer with a uniform distribution.

        The weights are initialized to be in the range [-weight_init, weight_init].
        """
        with torch.no_grad():
            # Initialize weights uniformly
            nn.init.uniform_(self.linear.weight, -self.weight_init, self.weight_init)

    def forward(self, x):
        """
        Applies the GaussianLayer to the input tensor.

        This applies the linear layer, then applies the Gaussian activation function to the output.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the GaussianLayer.
        """
        LOGGER.debug("FORWARD PASS")
        out = self.linear(x)
        return self.activation(out)

class GaussianResidualLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=30.0, weight_init=0.1):
        """
        Initializes a GaussianResidualLayer.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): Whether to include bias terms in the linear layer. Defaults to True.
            scale (float, optional): The scaling factor for the Gaussian activation function. Defaults to 30.0.
            weight_init (float, optional): The initialization value for the weights of the linear layer. Defaults to 0.1.
        """
        super().__init__()
        LOGGER.info("GAUSSIAN RESIDUAL LAYER")

        # Assign input and output feature dimensions
        self.in_features = in_features
        self.out_features = out_features

        # Set scaling factor and weight initialization factor
        self.scale = scale
        self.weight_init = weight_init

        # Initialize the Gaussian activation function
        self.activation = gaussian_activation(self.scale)

        # Define the linear transformation layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Initialize the weights of the linear layer
        self._init_weights()

    def _init_weights(self):
        """
        Initializes the weights of the linear layer with a uniform distribution.

        The weights are initialized to be in the range [-weight_init, weight_init].
        """
        with torch.no_grad():
            # Initialize weights uniformly
            nn.init.uniform_(self.linear.weight, -self.weight_init, self.weight_init)

    def forward(self, x):
        """
        Applies the GaussianResidualLayer to the input tensor.

        This applies the linear layer, then applies the Gaussian activation function to the output,
        and adds the input tensor to the output.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the GaussianResidualLayer.
        """
        LOGGER.debug("FORWARD PASS")
        out = self.linear(x)
        return self.activation(out) + x

class GaussianModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=128, hidden_layers=5, bias=True, final_bias=False, scale=30.0, weight_init=0.1, residual_net=False):
        """
        Initializes the GaussianModel with the specified parameters.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            hidden_features (int, optional): The number of features in the hidden layers. Defaults to 128.
            hidden_layers (int, optional): The number of hidden layers. Defaults to 5.
            bias (bool, optional): Whether to include bias terms in the linear layers. Defaults to True.
            final_bias (bool, optional): Whether to include bias terms in the final layer. Defaults to False.
            scale (float, optional): The scaling factor for the Gaussian activation function. Defaults to 30.0.
            weight_init (float, optional): The initialization value for the weights of the linear layers. Defaults to 0.1.
            residual_net (bool, optional): Whether to use residual connections in the network. Defaults to False.
        """
        super().__init__()
        LOGGER.info("Initializing Gaussian Model")

        # Initialize the network container
        self.net = []

        # Add the first Gaussian layer
        self.net.append(GaussianLayer(in_features, hidden_features, bias=bias, scale=scale, weight_init=weight_init))

        # Add hidden layers
        for i in range(hidden_layers):
            if residual_net:
                # Add a Gaussian residual layer if residual connections are enabled
                self.net.append(GaussianResidualLayer(hidden_features, hidden_features, bias=bias, scale=scale, weight_init=weight_init))
            else:
                # Add a standard Gaussian layer otherwise
                self.net.append(GaussianLayer(hidden_features, hidden_features, bias=bias, scale=scale, weight_init=weight_init))

        # Define and initialize the final linear layer
        final_layer = nn.Linear(hidden_features, out_features, bias=final_bias)
        with torch.no_grad():
            nn.init.uniform_(final_layer.weight, -weight_init, weight_init)
        self.net.append(final_layer)

        # Combine all layers into a sequential container
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        LOGGER.debug("FORWARD PASS")
        return self.net(x)


########################################################################################################################
##### GAUSSIAN FINER MODEL #############################################################################################
########################################################################################################################

class GaussianFinerLayer(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, omega_f=2.5, bias_init=1.0, weight_init=0.1):
        super().__init__()
        LOGGER.debug("GAUSSIAN FINER LAYER")

        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.omega_f = omega_f
        self.bias_init = bias_init
        self.weight_init = weight_init

        # Define a linear layer
        self.linear = nn.Linear(in_features, out_features, bias=True)

        # Initialize weights
        self._init_weights()
        self.activation = gaussian_activation(with_finer=True, scale=scale, omega_f=omega_f)


    def _init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-self.weight_init, self.weight_init)
            self.linear.bias.uniform_(-self.bias_init, self.bias_init)


    def forward(self, x):
        LOGGER.debug("FORWARD PASS")
        out = self.linear(x)
        return self.activation(out)



class GaussianFinerResidualLayer(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, omega_f=2.5, bias_init=1.0, weight_init=0.1):
        super().__init__()
        LOGGER.info("GAUSSIAN FINER RESIDUAL LAYER")

        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.omega_f = omega_f
        self.bias_init = bias_init
        self.weight_init = weight_init

        # Define a linear layer
        self.linear = nn.Linear(in_features, out_features, bias=True)

        # Initialize weights
        self._init_weights()
        self.activation = gaussian_activation(with_finer=True, scale=scale, omega_f=omega_f)

    def _init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-self.weight_init, self.weight_init)
            self.linear.bias.uniform_(-self.bias_init, self.bias_init)

    def forward(self, x):
        LOGGER.debug("FORWARD PASS")
        out = self.linear(x)
        return self.activation(out) + x

class GaussianFinerModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=128, hidden_layers=5, final_bias=False, scale=30.0, omega_f=2.5, bias_init=1.0, weight_init=0.1, residual_net=False):
        super().__init__()
        LOGGER.info("Initializing Gaussian Finer Model")

        # Initialize the network container
        self.net = []

        # Add the first Gaussian layer
        self.net.append(GaussianFinerLayer(in_features, hidden_features, scale=scale, omega_f=omega_f, bias_init=bias_init, weight_init=weight_init))

        # Add hidden layers
        for i in range(hidden_layers):
            if residual_net:
                # Add a Gaussian residual layer if residual connections are enabled
                self.net.append(GaussianFinerResidualLayer(hidden_features, hidden_features, scale=scale, omega_f=omega_f, bias_init=bias_init, weight_init=weight_init))
            else:
                # Add a standard Gaussian layer otherwise
                self.net.append(GaussianFinerLayer(hidden_features, hidden_features, scale=scale, omega_f=omega_f, bias_init=bias_init, weight_init=weight_init))

        # Define and initialize the final linear layer
        final_layer = nn.Linear(hidden_features, out_features, bias=final_bias)
        with torch.no_grad():
            nn.init.uniform_(final_layer.weight, -weight_init, weight_init)
        self.net.append(final_layer)

        # Combine all layers into a sequential container
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        LOGGER.debug("FORWARD PASS")
        return self.net(x)