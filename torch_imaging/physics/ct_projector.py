import torch

pi = 3.1415927410125732
from ..linalg.linear_operator import CompositeLinearOperator
from ..linalg.permute import Pad
from ..linalg.fourier import FourierTransform
from ..linalg.polar import PolarCoordinateResampler

class CTProjector_FourierSliceTheorem(CompositeLinearOperator):
    def __init__(self, input_shape, num_fourier_angular_samples, num_fourier_radial_samples, theta_values=None, radius_values=None):

        if theta_values is None:
            theta_values = torch.linspace(0, 2 * pi, num_fourier_angular_samples)
        if radius_values is None:
            # we use 3x zero padding so this is the spacing in resulting fourier space
            sample_spacing = 1.5*3.0*max(input_shape)/num_fourier_radial_samples 

            # Create two separate tensors for the negative and positive ranges
            negative_samples = torch.arange(start=-sample_spacing * (num_fourier_radial_samples//2), end=-sample_spacing, step=sample_spacing)
            positive_samples = torch.arange(start=0, end=sample_spacing * (num_fourier_radial_samples//2), step=sample_spacing)

            # Concatenate the tensors to get the final radius tensor
            radius_values = torch.cat([negative_samples, positive_samples], dim=0)
            
            # Store parameters
            self.num_row = input_shape[0]
            self.num_col = input_shape[1]
            self.num_fourier_angular_samples = num_fourier_angular_samples
            self.num_fourier_radial_samples = num_fourier_radial_samples
            self.theta_values = theta_values
            self.radius_values = radius_values

            # Create a pad operator to zero-pad the input image
            pad = Pad(input_shape, (input_shape[0], input_shape[0], input_shape[1], input_shape[1]))

            # Create 2D Fourier transform module
            fourier_transform_2d = FourierTransform((input_shape[0]*3, input_shape[1]*3), dim=(-2, -1))

            # Create Polar Coordinate Transformation module
            polar_transform = PolarCoordinateResampler((input_shape[0]*3, input_shape[1]*3), theta_values, radius_values)

            # Create 1D Fourier transform module
            fourier_transform_1d = FourierTransform((num_fourier_radial_samples,), dim=(-1,))

            operators = [pad, fourier_transform_2d, polar_transform, fourier_transform_1d.adjoint_LinearOperator()]

            super(CTProjector_FourierSliceTheorem, self).__init__(operators)
    