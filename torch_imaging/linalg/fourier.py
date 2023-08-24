
import torch

from .linear_operator import LinearOperator, InvertibleLinearOperator
    
class UnitaryFourierTransform(LinearOperator):
    def __init__(self, input_shape, dim):
        """
        This class implements a N-Dimensional Fourier transform that can be used in a PyTorch model.

        it assumes the central pixel in the image is at 0

        it returns the Fourier transform with the zero-frequency component in the center of the image
        """
        super(UnitaryFourierTransform, self).__init__(input_shape)
        self.dim = dim

    def forward(self, x):
        x_ifftshift = torch.fft.ifftshift(x, dim=self.dim)
        x_fft = torch.fft.fftn(x_ifftshift, dim=self.dim)
        x_fft = x_fft / torch.sqrt(torch.prod(torch.tensor([x.shape[d] for d in self.dim])))
        x_fftshift = torch.fft.fftshift(x_fft, dim=self.dim)
        return x_fftshift
    
    def adjoint(self, y):
        y_ifftshift = torch.fft.ifftshift(y, dim=self.dim)
        y_ifft = torch.fft.ifftn(y_ifftshift, dim=self.dim)
        y_ifft = y_ifft * torch.sqrt(torch.prod(torch.tensor([y.shape[d] for d in self.dim])))
        y_fftshift = torch.fft.fftshift(y_ifft, dim=self.dim)
        return y_fftshift

class FourierFilter2D(InvertibleLinearOperator):
    def __init__(self, input_shape, filter):
        """
        This class implements a 2D Fourier filter that can be used in a PyTorch model.

        it assumes the central pixel in the image is at (0,0)

        it returns the Fourier filter applied to the input. 
        """
        super(FourierFilter2D, self).__init__(input_shape, input_shape)
        self.fourier_transform_2d = UnitaryFourierTransform(input_shape, dim=(-2, -1))
        self.filter = filter
    def forward(self, x):
        x_fftshift = self.fourier_transform_2d.forward(x)
        x_filtered = x_fftshift * self.filter
        x_ifftshift = self.fourier_transform_2d.adjoint(x_filtered)
        return x_ifftshift
    
class FourierFilter1D(InvertibleLinearOperator):
    def __init__(self, input_shape, filter):
        """
        This class implements a 1D Fourier filter that can be used in a PyTorch model.

        it assumes the central pixel in the image is at (0,0)

        it returns the Fourier filter applied to the input
        """
        super(FourierFilter1D, self).__init__(input_shape, input_shape)
        self.fourier_transform_1d = UnitaryFourierTransform(input_shape, dim=(-1,))
        self.filter = filter
    def forward(self, x):
        x_fft = self.fourier_transform_1d.forward(x)
        x_filtered = x_fft * self.filter
        x_convolved = self.fourier_transform_1d.adjoint(x_filtered)
        return x_convolved
    
class FourierConvolution2D(InvertibleLinearOperator):
    def __init__(self, input_shape, kernel):
        """
        This class implements a 2D Fourier convolution that can be used in a PyTorch model.

        it assumes the central pixel in the image is at (0,0), including for the input kernel

        it returns the Fourier transform with the zero-frequency component in the center of the image
        """
        super(FourierConvolution2D, self).__init__(input_shape, input_shape)
        self.fourier_transform_2d = UnitaryFourierTransform(input_shape, dim=(-2, -1))
        self.kernel = kernel
    def forward(self, x):
        x_fft = self.fourier_transform_2d.forward(x)
        x_filtered = x_fft * self.kernel
        x_convolved = self.fourier_transform_2d.adjoint(x_filtered)
        return x_convolved
    
class FourierConvolution1D(InvertibleLinearOperator):
    def __init__(self, input_shape, kernel):
        """
        This class implements a 1D Fourier convolution that can be used in a PyTorch model.

        it assumes the central pixel in the image is at (0,0), including for the input kernel

        it returns the Fourier transform with the zero-frequency component in the center of the image
        """
        super(FourierConvolution1D, self).__init__(input_shape, input_shape)
        self.fourier_transform_1d = UnitaryFourierTransform(input_shape, dim=(-1,))
        self.kernel = kernel
    def forward(self, x):
        x_fft = self.fourier_transform_1d.forward(x)
        x_filtered = x_fft * self.kernel
        x_convolved = self.fourier_transform_1d.adjoint(x_filtered)
        return x_convolved
    
    

