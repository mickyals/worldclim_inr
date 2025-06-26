# define siren layer
# define siren model
# define logic for residual layer

import torch
import torch.nn as nn
import numpy
import torch.nn.functional as F
from helpers import get_logger
from models.implicit_neural_representations.activations import siren_activation

LOGGER = get_logger(name = "SirenModel", log_file="worldclim-dataset.log")

class SirenLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.):
        """
        Initializes the SirenLayer with the given parameters.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): Whether to include bias terms in the linear layer. Defaults to True.
            is_first (bool, optional): Indicates if this is the first layer. Defaults to False.
            omega_0 (float, optional): The frequency scaling factor. Defaults to 30.
        """
        super().__init__()
        LOGGER.info("Siren Layer")
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features
        self.activation = siren_activation(self.omega_0)

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self):
        """
        Initializes the weights of the linear layer.

        Depending on whether the layer is the first in the network or a hidden layer,
        it applies a different initialization strategy to the weights.

        Logging is used to trace the initialization process.
        """
        LOGGER.info("Initializing weights")
        with torch.no_grad():
            if self.is_first:
                # Initialize the weights for the first layer with a uniform distribution
                LOGGER.info("Initializing first weights")
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                # Initialize the weights for hidden layers with a scaled uniform distribution
                LOGGER.info("Initializing hidden weights")
                bound = numpy.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)


    def forward(self, x):
        """
        Applies the SirenLayer to the input tensor.

        This applies the linear layer, then applies the siren activation function to the output.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the SirenLayer.
        """
        LOGGER.info("SIREN LAYER")
        # Apply the linear layer
        out = self.linear(x)
        # Apply the siren activation function
        return self.activation(out)



class SirenResidualLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega_0=30.):
        """
        Initializes the SirenResidualLayer.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): Whether to include bias terms in the linear layer. Defaults to True.
            omega_0 (float, optional): The frequency scaling factor. Defaults to 30.
        """
        super().__init__()
        LOGGER.info("SIREN RESIDUAL LAYER")
        self.omega_0 = omega_0
        self.in_features = in_features
        self.out_features = out_features
        self.activation = siren_activation(self.omega_0)

        # Define a linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initializes the weights of the linear layer using the uniform distribution.

        The weights are initialized to be in the range [-bound, bound].
        """
        LOGGER.info("Initialzing weights")
        with torch.no_grad():
            # Calculate the bound for the uniform distribution
            bound = numpy.sqrt(6/self.in_features)/self.omega_0
            self.linear.weight.uniform_(-bound, bound)


    def forward(self, x):
        """
        Perform the forward pass for the SirenResidualLayer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the SirenResidualLayer with residual connection.
        """
        LOGGER.info("SIREN RESIDUAL LAYER")
        # Apply the linear layer and siren activation, then add the input tensor for residual connection
        out = self.linear(x)
        return self.activation(out) + x



class SirenModel(nn.Module):
    def __init__(self, in_features=2, out_features=2, hidden_layers=5, hidden_features=128, first_omega_0=30., hidden_omega_0=30., residual_net=False):
        """
        Initializes a SirenModel with the given parameters.

        Args:
            in_features (int, optional): The number of input features. Defaults to 2.
            out_features (int, optional): The number of output features. Defaults to 2.
            hidden_layers (int, optional): The number of hidden layers. Defaults to 5.
            hidden_features (int, optional): The number of features in the hidden layers. Defaults to 128.
            first_omega_0 (float, optional): The frequency scaling factor for the first layer. Defaults to 30.
            hidden_omega_0 (float, optional): The frequency scaling factor for the hidden layers. Defaults to 30.
            residual_net (bool, optional): Whether to use residual connections. Defaults to False.
        """
        super().__init__()
        LOGGER.info("SIREN MODEL")
        self.net = []
        # build the first layer
        self.net.append(SirenLayer(in_features, hidden_features, bias=True, is_first=True, omega_0=first_omega_0))

        # build the hidden layers
        for i in range(hidden_layers):
            if residual_net:
                # build a residual layer
                LOGGER.info(f"SIREN RESIDUAL LAYER {i}")
                self.net.append(SirenResidualLayer(hidden_features, hidden_features, bias=True, omega_0=hidden_omega_0))
            else:
                # build a normal layer
                LOGGER.info(f"SIREN LAYER {i}")
                self.net.append(SirenLayer(hidden_features, hidden_features, bias=True, is_first=False, omega_0=hidden_omega_0))

        # build the final layer
        final_layer = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_layer.weight.uniform_(-numpy.sqrt(6/hidden_features)/hidden_omega_0, numpy.sqrt(6/hidden_features)/hidden_omega_0)
        self.net.append(final_layer)
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """
        Execute the forward pass through the SirenModel.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the model's network.
        """
        LOGGER.info("FORWARDING SIREN MODEL")
        # Forward the input tensor through the network
        return self.net(x)



###### FINER LAYERS


class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega_0=30.0, is_first=False, is_last=False, first_bias=None, hidden_bias=None):
        """
        Initializes a layer with configurable features and biases.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): Whether to include bias terms in the linear layer. Defaults to True.
            omega_0 (float, optional): The frequency scaling factor. Defaults to 30.0.
            is_first (bool, optional): Indicates if this is the first layer. Defaults to False.
            is_last (bool, optional): Indicates if this is the last layer. Defaults to False.
            first_bias (float, optional): The bias for the first layer. Defaults to None.
            hidden_bias (float, optional): The bias for hidden layers. Defaults to None.
        """
        super().__init__()

        # Assign input and output features
        self.in_features = in_features
        self.out_features = out_features
        # Frequency scaling factor
        self.omega_0 = omega_0
        # Flags for first and last layers
        self.is_first = is_first
        self.is_last = is_last
        # Bias values
        self.first_bias = first_bias
        self.hidden_bias = hidden_bias
        self.activation = siren_activation(self.omega_0, with_finer=True)

        # Define a linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Initialize weights
        self._init_weights()
        # Initialize bias if applicable
        if bias:
            self._init_bias()

    def _init_weights(self):
        """
        Initializes the weights of the linear layer.
        """
        with torch.no_grad():
            # Initialize the weights of the first layer
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                # Initialize the weights of the hidden layers
                bound = numpy.sqrt(6/self.in_features)/self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def _init_bias(self):
        """
        Initializes the bias of the linear layer.
        """
        with torch.no_grad():
            if self.is_first and self.first_bias is not None:
                # Initialize the bias of the first layer
                self.linear.bias.uniform_(-self.first_bias, self.first_bias)
            elif not self.is_first and self.hidden_bias is not None:
                # Initialize the bias of the hidden layers
                self.linear.bias.uniform_(-self.hidden_bias, self.hidden_bias)

    def forward(self, x):
        """
        Applies the FinerLayer to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the FinerLayer.
        """
        LOGGER.info("FINER LAYER")
        # Apply the linear layer
        out = self.linear(x)
        # Apply the siren activation function
        return self.activation(out)

class FinerResidualLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega_0=30.0, hidden_bias=None):
        """
        Initializes a FinerResidualLayer.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): Whether to include bias terms in the linear layers. Defaults to True.
            omega_0 (float, optional): The frequency scaling factor. Defaults to 30.0.
            hidden_bias (float, optional): The bias for the hidden layers. Defaults to None.
        """
        super().__init__()

        LOGGER.info("BUILDING FINER RESIDUAL LAYER")
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0
        self.hidden_bias = hidden_bias
        self.activation = siren_activation(self.omega_0, with_finer=True)

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()
        if bias and hidden_bias is not None:
            self._init_bias()

    def _init_weights(self):
        """
        Initializes the weights of the linear layer.
        """
        LOGGER.info("FINER RESIDUAL LAYER")
        with torch.no_grad():
            bound = numpy.sqrt(6/self.in_features)/self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def _init_bias(self):
        """
        Initializes the bias of the linear layer.
        """
        LOGGER.info("initializing bias")
        with torch.no_grad():
            self.linear.bias.uniform_(-self.hidden_bias, self.hidden_bias)

    def forward(self, x):
        """
        Forward passes through the FinerResidualLayer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the FinerResidualLayer.
        """
        LOGGER.info("FINER RESIDUAL LAYER")
        out = self.linear(x)
        return self.activation(out) + x

class FinerModel(nn.Module):
    def __init__(self, in_features=2, out_features=2, bias=True, hidden_layers=5, hidden_features=128, first_omega_0=30., hidden_omega_0=30.,
                 first_bias=None, hidden_bias=None, residual_net=False):
        """
        Initializes the FinerModel.

        Args:
            in_features (int, optional): The number of input features. Defaults to 2.
            out_features (int, optional): The number of output features. Defaults to 2.
            bias (bool, optional): Whether to include bias terms in the linear layers. Defaults to True.
            hidden_layers (int, optional): The number of hidden layers. Defaults to 5.
            hidden_features (int, optional): The number of features in the hidden layers. Defaults to 128.
            first_omega_0 (float, optional): The frequency scaling factor for the first layer. Defaults to 30.
            hidden_omega_0 (float, optional): The frequency scaling factor for the hidden layers. Defaults to 30.
            first_bias (float, optional): The maximum absolute value of the bias for the first layer. Defaults to None.
            hidden_bias (float, optional): The maximum absolute value of the bias for the hidden layers. Defaults to None.
            residual_net (bool, optional): Whether to use residual connections. Defaults to False.
        """
        super().__init__()

        LOGGER.info("INITIALIZING FINER MODEL")
        self.net = []
        self.net.append(FinerLayer(in_features, hidden_features, bias=bias, is_first=True, omega_0=first_omega_0, first_bias=first_bias))

        for i in range(hidden_layers):
            # build the hidden layers
            LOGGER.info("BUILDING FINER MODEL")
            if residual_net:
                LOGGER.info(f"RESIDUAL LAYER {i}")
                self.net.append(FinerResidualLayer(hidden_features, hidden_features, bias=True, omega_0=hidden_omega_0, hidden_bias=hidden_bias))
            else:
                LOGGER.info(f"NORMAL LAYER {i}")
                self.net.append(FinerLayer(hidden_features, hidden_features, bias=True, is_first=False, omega_0=hidden_omega_0, hidden_bias=hidden_bias))

        # build the final layer
        final_layer = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_layer.weight.uniform_(-numpy.sqrt(6/hidden_features)/hidden_omega_0, numpy.sqrt(6/hidden_features)/hidden_omega_0)
        self.net.append(final_layer)
        self.net = nn.Sequential(*self.net)


    def forward(self, x):
        """
        Forward pass through the FinerModel.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        LOGGER.info("FORWARDING FINER MODEL")
        return self.net(x)