# define gaussian layer
# define gaussian model
# define logic for residual layer

import torch
import torch.nn as nn
import numpy
from helpers import get_logger
from utils.activations import gaussian_activation
from utils.Initializers import Initializer
from utils.positional_encodings import ENCODER_REGISTRY

LOGGER = get_logger(name = "GaussianModel", log_file="worldclim-dataset.log")


class GaussianLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=30.0, weight_init=0.1, bias_init=None, dropout_prc=0.0):
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
        self.bias_init = bias_init
        self.activation = gaussian_activation(self.scale)
        self.dropout = nn.Dropout(dropout_prc) if dropout_prc > 0.0 else nn.Identity()

        # Define the linear layer using nn.Linear
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Initialize the weights of the linear layer
        Initializer.gaussian_init(self.linear, self.weight_init, self.bias_init, bias)


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
        out = self.activation(out)
        return self.dropout(out)

class GaussianResidualLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=30.0, weight_init=0.1, bias_init=None, dropout_prc=0.0):
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
        self.bias_init = bias_init

        # Initialize the Gaussian activation function
        self.activation = gaussian_activation(self.scale)
        self.dropout = nn.Dropout(dropout_prc) if dropout_prc > 0.0 else nn.Identity()

        # Define the linear transformation layer
        self.first_linear = nn.Linear(in_features, out_features, bias=bias)
        self.last_linear = nn.Linear(out_features, out_features, bias=bias)

        # Initialize the weights of the linear layer
        Initializer.gaussian_init(self.first_linear, self.weight_init, self.bias_init, bias)
        Initializer.gaussian_init(self.last_linear, self.weight_init, self.bias_init, bias)

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
        out = self.first_linear(x)
        first_activation = self.activation(out)
        out = self.last_linear(first_activation)
        out = out + x
        out = self.activation(out)
        return self.dropout(out)

class GaussianModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=128, hidden_layers=5,
                 bias=True, final_bias=False, scale_gaussian=30.0, weight_init=0.1, bias_init=None,
                 residual_net=False, dropout=0.0, encoding=None, **encoder_kwargs):
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

        # Add the first Gaussian layer
        self.net.append(GaussianLayer(encoded_dim, hidden_features, bias=bias, scale=scale_gaussian, weight_init=weight_init, bias_init=bias_init, dropout_prc=0.0))

        # Add hidden layers
        for i in range(hidden_layers):
            if residual_net:
                # Add a Gaussian residual layer if residual connections are enabled
                self.net.append(GaussianResidualLayer(hidden_features, hidden_features, bias=bias, scale=scale_gaussian, weight_init=weight_init, bias_init=bias_init, dropout_prc=dropout))
            else:
                # Add a standard Gaussian layer otherwise
                self.net.append(GaussianLayer(hidden_features, hidden_features, bias=bias, scale=scale_gaussian, weight_init=weight_init, bias_init=bias_init, dropout_prc=dropout))

        # Define and initialize the final linear layer
        final_layer = nn.Linear(hidden_features, out_features, bias=final_bias)
        with torch.no_grad():
            nn.init.uniform_(final_layer.weight, -weight_init, weight_init)
        self.net.append(final_layer)

        # Combine all layers into a sequential container
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        LOGGER.debug("FORWARD PASS")
        x = self.encoder(x)
        return self.net(x)


########################################################################################################################
##### GAUSSIAN FINER MODEL #############################################################################################
########################################################################################################################

class GaussianFinerLayer(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, omega_f=2.5,
                 first_k=10, hidden_k=10, is_first=False, dropout_prc=0.0):
        super().__init__()
        LOGGER.debug("GAUSSIAN FINER LAYER")

        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.omega_f = omega_f
        self.first_k = first_k
        self.hidden_k = hidden_k
        self.is_first = is_first

        # Define a linear layer
        self.linear = nn.Linear(in_features, out_features, bias=True)


        # Initialize weights
        Initializer.gaussian_finer_init(self.linear, self.in_features, self.omega_f, first_k=self.first_k, hidden_k=self.hidden_k)
        self.activation = gaussian_activation(with_finer=True, scale=scale, omega_f=omega_f)
        self.dropout = nn.Dropout(dropout_prc) if dropout_prc > 0.0 else nn.Identity()



    def forward(self, x):
        LOGGER.debug("FORWARD PASS")
        out = self.linear(x)
        out = self.activation(out)
        return self.dropout(out)



class GaussianFinerResidualLayer(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, omega_f=2.5, first_k=10, hidden_k=10, dropout_prc=0.0):
        super().__init__()
        LOGGER.info("GAUSSIAN FINER RESIDUAL LAYER")

        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.omega_f = omega_f
        self.first_k = first_k
        self.hidden_k = hidden_k
        # Define a linear layer
        self.first_linear = nn.Linear(in_features, out_features, bias=True)
        self.last_linear = nn.Linear(out_features, out_features, bias=True)

        # Initialize weights
        Initializer.gaussian_finer_init(self.first_linear, self.in_features, self.omega_f, first_k=self.first_k, hidden_k=self.hidden_k)
        Initializer.gaussian_finer_init(self.last_linear, self.out_features, self.omega_f, first_k=self.first_k, hidden_k=self.hidden_k)
        self.activation = gaussian_activation(with_finer=True, scale=scale, omega_f=omega_f)
        self.dropout = nn.Dropout(dropout_prc) if dropout_prc > 0.0 else nn.Identity()

    def forward(self, x):
        LOGGER.debug("FORWARD PASS")
        out = self.first_linear(x)
        first_activation = self.activation(out)
        out = self.last_linear(first_activation)
        out = out + x
        out = self.activation(out)
        return self.dropout(out)

class GaussianFinerModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=128, hidden_layers=5,
                 final_bias=False, scale_gaussian=30.0, omega_f=2.5, first_k=10, hidden_k=10, residual_net=False,
                 dropout=0.0, encoding=None, **encoder_kwargs):
        super().__init__()
        LOGGER.info("Initializing Gaussian Finer Model")

        # Initialize the network container
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

        self.net.append(GaussianFinerLayer(encoded_dim, hidden_features, scale=scale_gaussian, omega_f=omega_f,
                                               first_k=first_k, hidden_k=hidden_k, is_first=True, dropout_prc = 0.0))

        # Add hidden layers
        for i in range(hidden_layers):
            if residual_net:
                # Add a Gaussian residual layer if residual connections are enabled
                self.net.append(GaussianFinerResidualLayer(hidden_features, hidden_features, scale=scale_gaussian,
                                                           omega_f=omega_f, first_k=first_k, hidden_k=hidden_k, dropout_prc=dropout))
            else:
                # Add a standard Gaussian layer otherwise
                self.net.append(GaussianFinerLayer(hidden_features, hidden_features, scale=scale_gaussian, omega_f=omega_f,
                                                   first_k=first_k, hidden_k=hidden_k, dropout_prc=dropout))

        # Define and initialize the final linear layer
        final_layer = nn.Linear(hidden_features, out_features, bias=final_bias)
        with torch.no_grad():
            bound = numpy.sqrt(6 / hidden_features) / omega_f
            nn.init.uniform_(final_layer.weight, -bound, bound)
        self.net.append(final_layer)

        # Combine all layers into a sequential container
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        LOGGER.debug("FORWARD PASS")
        x = self.encoder(x)
        return self.net(x)