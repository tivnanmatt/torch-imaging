import torch

pi = 3.1415927410125732

from .polar import PolarCoordinateResampler
from .fourier import UnitaryFourierTransform



class CTProjector_FourierSliceTheorem(torch.nn.Module):
    def __init__(self, input_shape, num_fourier_angular_samples, num_fourier_radial_samples, theta_values=None, radius_values=None):
        super(CTProjector_FourierSliceTheorem, self).__init__()

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

            # Create 2D Fourier transform module
            self.fourier_transform_2d = UnitaryFourierTransform((input_shape[0]*3, input_shape[1]*3), dim=(-2, -1))

            # Create Polar Coordinate Transformation module
            self.polar_transform = PolarCoordinateResampler((input_shape[0]*3, input_shape[1]*3), theta_values, radius_values)

            # Create 1D Fourier transform module
            self.fourier_transform_1d = UnitaryFourierTransform((num_fourier_radial_samples,), dim=(-1,))

    def forward(self, x):
        # Zero-pad the input image
        x_padded = torch.nn.functional.pad(x, (x.shape[-2], x.shape[-2], x.shape[-1], x.shape[-1]))

        # 2D Fourier transform
        x_fft2 = self.fourier_transform_2d.forward(x_padded)
        
        # Polar coordinate transformation
        projections_fft = self.polar_transform(x_fft2)
        
        # 1D Fourier transform along the radial direction
        projections = self.fourier_transform_1d.adjoint(projections_fft)

        return projections

    def inverse(self, y, max_iter=50, verbose=False):

        # invert the forward step by step
        y = torch.fft.ifftshift(y, dim=-1)
        y = torch.fft.fft(y, dim=-1)
        y = torch.fft.fftshift(y, dim=-1)

        # inverse the polar coordinate transformation
        x_fft2 = self.polar_transform.inverse(y, max_iter=max_iter, verbose=verbose)

        # inverse the 2D Fourier transform
        x_fft2 = torch.fft.ifftshift(x_fft2, dim=(-2, -1))
        x = torch.fft.ifft2(x_fft2, dim=(-2, -1))
        x = torch.fft.fftshift(x, dim=(-2, -1))

        # crop back to original size
        x = x[:, :, self.num_row:2*self.num_row, self.num_col:2*self.num_col]

        return x
    
    def to(self, device):
        self.polar_transform.to(device)
        return super(CTProjector_FourierSliceTheorem, self).to(device)
    