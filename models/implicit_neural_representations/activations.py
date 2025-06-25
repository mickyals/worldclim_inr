# define activation functions for implicit neural representations

import torch
import torch.nn as nn
import torch.nn.functional as F

class siren_activation(nn.Module):
    def __init__(self, omega=1, with_finer=False):
        """
        Initializes the activation function with the given parameters.

        Args:
            omega (float, optional): The frequency scaling factor. Defaults to 1.
            with_finer (bool, optional): Whether to apply finer adjustments. Defaults to False.
        """
        super().__init__()
        # Set the frequency scaling factor
        self.omega = omega
        # Determine if finer adjustments should be applied
        self.with_finer = with_finer

    @staticmethod
    def generate_alpha(x):
        """
        Generates the alpha value for the siren activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The alpha value.
        """
        with torch.no_grad():
            # The formula for alpha is |x| + 1
            return torch.abs(x) + 1

    def forward(self, x):
        """
        Applies the siren activation function to the input tensor.

        If `with_finer` is True, then the function applies finer adjustments to the input tensor.
        Otherwise, it applies the standard sinc activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the sinc activation.
        """
        if self.with_finer:
            # Generate alpha for finer adjustments
            alpha = self.generate_alpha(x)
            # Modulate x with omega and alpha
            mod_x = alpha * x
        else:
            # Modulate x with omega
            mod_x = self.omega * x
        # Apply sine function to the modulated x
        return torch.sin(mod_x)


class gaussian_activation(nn.Module):
    def __init__(self, scale, with_finer=False, omega=1):
        """
        Initializes the activation function with the given parameters.

        Args:
            scale (float): The scaling factor for the activation function.
            with_finer (bool, optional): Whether to apply finer adjustments. Defaults to False.
            omega (float, optional): The frequency scaling factor. Defaults to 1.
        """
        super().__init__()
        # Set the scaling factor for the activation function
        self.scale = scale
        # Determine if finer adjustments should be applied
        self.with_finer = with_finer
        # Set the frequency scaling factor
        self.omega = omega

    @staticmethod
    def generate_alpha(x):
        """
        Generates the alpha value for the Gaussian activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The alpha value.
        """
        with torch.no_grad():
            # The formula for alpha is |x| + 1
            return torch.abs(x) + 1

    def forward(self, x):
        """
        Applies the Gaussian activation function to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the Gaussian activation.
        """
        if self.with_finer:
            # Generate alpha for finer adjustments
            alpha = self.generate_alpha(x)
            # Modulate x with omega and alpha
            x = torch.sin(self.omega * alpha * x)

        # Apply the Gaussian activation function
        return torch.exp(-(self.scale * x)**2)


class wire_activation(nn.Module):
    def __init__(self, scale, omega_w, with_finer=False, omega=1 ):
        """
        Initializes the wire activation function with the given parameters.

        Args:
            scale (float): The scaling factor for the wire activation function.
            omega_w (float): The frequency scaling factor for the wire activation function.
            with_finer (bool, optional): Whether to apply finer adjustments. Defaults to False.
            omega (float, optional): The frequency scaling factor for the finer adjustments. Defaults to 1.
        """
        super().__init__()
        self.scale = scale
        self.omega_w= omega_w
        self.with_finer = with_finer
        self.omega = omega

    @staticmethod
    def generate_alpha(x):
        """
        Generates the alpha value for the wire activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The alpha value.
        """
        with torch.no_grad():
            return torch.abs(x) + 1

    @staticmethod
    def generate_complex_alpha(x):
        """
        Generates the complex alpha value for the wire activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The complex alpha value.
        """
        with torch.no_grad():
            # Calculate the absolute value of the real and imaginary parts of x
            alpha_real = torch.abs(x.real) + 1
            alpha_imag = torch.abs(x.imag) + 1

        # Scale the real and imaginary parts of x by the corresponding alpha values
        real_scaled = x.real * alpha_real
        imag_scaled = x.imag * alpha_imag

        # Combine the scaled real and imaginary parts to form the complex alpha value
        return torch.complex(real_scaled, imag_scaled)

    def forward(self, x):
        """
        Applies the wire activation function to the input tensor.

        If `with_finer` is True, then the function applies finer adjustments to the input tensor.
        Otherwise, it applies the standard wire activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the wire activation.
        """
        if self.with_finer:
            if x.is_complex():
                # Generate the complex alpha value for the wire activation function
                alpha_complex = self.generate_complex_alpha(x)
                # Apply the wire activation function with the complex alpha value
                mod_x = torch.sin(self.omega * alpha_complex)
            else:
                # Generate the alpha value for the wire activation function
                alpha = self.generate_alpha(x)
                # Apply the wire activation function with the alpha value
                mod_x = torch.sin(self.omega * alpha * x)
        else:
            # Apply the standard wire activation function
            mod_x = x
        # Apply the exponential function to the output of the wire activation function
        return torch.exp((1j * self.omega_w * mod_x) - torch.abs(self.scale * mod_x)**2)




class hosc_activation(nn.Module):
    def __init__(self, beta, omega=1, with_finer=False):
        """
        Initializes the hosc_activation function with the given parameters.

        Args:
            beta (float): The scaling factor for the tanh function.
            omega (float, optional): The frequency scaling factor. Defaults to 1.
            with_finer (bool, optional): Whether to apply finer adjustments. Defaults to False.
        """
        super().__init__()
        # Set the scaling factor for the tanh function
        self.beta = beta
        # Determine if finer adjustments should be applied
        self.with_finer = with_finer
        # Set the frequency scaling factor
        self.omega = omega

    @staticmethod
    def generate_alpha(x):
        """
        Generates the alpha value for the hosc_activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The alpha value.
        """
        with torch.no_grad():
            # The formula for alpha is |x| + 1
            return torch.abs(x) + 1

    def forward(self, x):
        """
        Applies the hosc_activation function to the input tensor.

        If `with_finer` is True, then the function applies finer adjustments to the input tensor.
        Otherwise, it applies the standard sinc activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the sinc activation.
        """
        if self.with_finer:
            # Generate the alpha value for the finer adjustments
            alpha = self.generate_alpha(x)
            # Apply the finer adjustments
            mod_x = self.omega * alpha * x
        else:
            # Apply the standard sinc activation function
            mod_x = self.omega * x
        # Apply the tanh function with the given scaling factor
        return torch.tanh(self.beta * torch.sin(mod_x))


class sinc_activation(nn.Module):
    def __init__(self, omega=1, with_finer=False):
        """
        Initializes the activation function with the given parameters.

        Args:
            omega (float, optional): The frequency scaling factor. Defaults to 1.
            with_finer (bool, optional): Whether to apply finer adjustments. Defaults to False.
        """
        super().__init__()
        # Set the frequency scaling factor
        self.omega = omega
        # Determine if finer adjustments should be applied
        self.with_finer = with_finer

    @staticmethod
    def generate_alpha(x):
        """
        Generates the alpha value for the sinc activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The alpha value.
        """
        with torch.no_grad():
            # The formula for alpha is |x| + 1
            return torch.abs(x) + 1

    def forward(self, x):
        """
        Applies the sinc activation function to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the sinc activation.
        """
        if self.with_finer:
            # Generate alpha for finer adjustments
            alpha = self.generate_alpha(x)
            # Modulate x with omega and alpha
            mod_x = self.omega * alpha * x
        else:
            # Modulate x with omega
            mod_x = self.omega * x
        # Apply sinc function to the modulated x
        return torch.sinc(mod_x)

