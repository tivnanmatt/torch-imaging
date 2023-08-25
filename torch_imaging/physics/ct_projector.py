import torch

pi = 3.1415927410125732
from ..linalg.linear_operator import CompositeLinearOperator
from ..linalg.permute import Pad
from ..linalg.fourier import FourierTransform
from ..linalg.polar import PolarCoordinateResampler

from ..linalg.sparse import SparseLinearOperator

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






class CTProjector_ParallelBeam2D(SparseLinearOperator):

    def __init__(self, Nx, Ny, dx, dy, Nu, du, theta, verbose=False):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.Nu = Nu
        self.du = du
        self.theta = theta
        self.Ntheta = len(theta)
        self.verbose = verbose

        # Precompute indices and weights
        indices, weights = self._precompute_weights_and_indices()

        # Now we initialize the parent class with the indices and weights
        super(CTProjector_ParallelBeam2D, self).__init__((Nx, Ny), (len(theta), Nu), indices, weights)
    
    def _precompute_weights_and_indices(self):
        all_indices = []
        all_weights = []

        y = torch.linspace(-(self.Ny-1)/2, (self.Ny-1)/2, self.Ny) * self.dy
        x = torch.linspace(-(self.Nx-1)/2, (self.Nx-1)/2, self.Nx) * self.dx
        y2d, x2d = torch.meshgrid(y, x, indexing='ij')

        for iTheta, theta_i in enumerate(self.theta):
            pixel_index, area_between_pixel_trapezoidal_footprint = self._system_response(theta_i, y2d, x2d)
            num_nonzero_pixels = pixel_index.shape[0]

            # Reshaping to accumulate for all thetas
            pixel_index_vectors = pixel_index.reshape([num_nonzero_pixels, self.Nx*self.Ny])
            area_overlap_vectors = area_between_pixel_trapezoidal_footprint.reshape([num_nonzero_pixels, self.Nx*self.Ny])

            all_indices.append(pixel_index_vectors)
            all_weights.append(area_overlap_vectors)

        # Concatenate all indices and weights from different views into tensors
        all_indices = torch.cat(all_indices, dim=0)
        all_weights = torch.cat(all_weights, dim=0)

        return all_indices, all_weights
    
    def _system_response(self, theta_i, y2d, x2d):

        # compute the projection
        theta_i = torch.tensor(theta_i)

        # absolute value of the sine and cosine of the angle
        abs_cos_theta = torch.abs(torch.cos(theta_i))
        abs_sin_theta = torch.abs(torch.sin(theta_i))

        # height of the trapezoid
        h = torch.minimum(self.dy/abs_cos_theta, self.dx/abs_sin_theta)
        # base 1 of the trapezoid
        b1 = torch.abs(self.dy*abs_sin_theta - self.dx*abs_cos_theta)
        # base 2 of the trapezoid
        b2 = torch.abs(self.dy*abs_sin_theta + self.dx*abs_cos_theta)

        # below depends on x and y, above depends only on theta

        # center of the trapezoid
        u0 = (x2d*torch.cos(theta_i) + y2d*torch.sin(theta_i))
        # left edge of the trapezoid base 2 
        u1 = u0 - b2/2
        # left edge of the trapezoid base 1
        u2 = u0 - b1/2
        # right edge of the trapezoid base 1
        u3 = u0 + b1/2
        # right edge of the trapezoid base 2
        u4 = u0 + b2/2

        # compute the index of the projection
        u1_index = self._convert_u_to_projection_index(u1)
        u4_index = self._convert_u_to_projection_index(u4)
        num_nonzero_pixels = int(torch.max(torch.ceil(u4_index)-torch.floor(u1_index))) + 1
        pixel_index = torch.zeros([num_nonzero_pixels, x2d.shape[0], x2d.shape[1]], dtype=torch.long).to(device)
        area_between_pixel_trapezoidal_footprint = torch.zeros([num_nonzero_pixels, x2d.shape[0], x2d.shape[1]], dtype=torch.float).to(device)

        for iPixel in range(num_nonzero_pixels):
            # get the index of the pixel of interest
            pixel_index[iPixel] = torch.floor(u1_index).long() + iPixel
            # convert index to u coordinate
            u = self._convert_projection_index_to_u(pixel_index[iPixel])
            # area of the left side of the trapezoid
            u_A = torch.maximum(u1,u-self.du/2)
            u_B = torch.minimum(u2,u+self.du/2)
            area_between_pixel_trapezoidal_footprint[iPixel] += (u_B>u_A)*(h/(2*(u2-u1 + (u1==u2))))*((u_B-u1)**2.0 - (u_A-u1)**2.0)
            # area of the center of the trapezoidm
            u_A = torch.maximum(u2,u-self.du/2)
            u_B = torch.minimum(u3,u+self.du/2)
            area_between_pixel_trapezoidal_footprint[iPixel] += (u_B>u_A)*h*(u_B-u_A)
            # area of the right side of the trapezoid
            u_A = torch.maximum(u3,u-self.du/2)
            u_B = torch.minimum(u4,u+self.du/2)
            area_between_pixel_trapezoidal_footprint[iPixel] += (u_B>u_A)*(h/(2*(u4-u3+ (u3==u4))))*((u_A-u4)**2.0 - (u_B-u4)**2.0)

        return pixel_index, area_between_pixel_trapezoidal_footprint 
    