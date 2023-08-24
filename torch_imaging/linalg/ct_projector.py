import torch

from .polar import PolarCoordinateResampler



class CTProjector_FourierSliceTheorem(torch.nn.Module):
    def __init__(self, num_row, num_col, theta_values, radius_values):
        super(CTProjector_FourierSliceTheorem, self).__init__()
        
        # Store parameters
        self.num_row = num_row
        self.num_col = num_col

        # Create Polar Coordinate Transformation module
        self.polar_transform = PolarCoordinateResampler((num_row, num_col), theta_values, radius_values)

    def forward(self, x):
        # Zero-pad the input image
        x_padded = torch.nn.functional.pad(x, (x.shape[-1], x.shape[-1], x.shape[-2], x.shape[-2]))

        # Compute 2D Fourier transform on the padded image
        x_fft2 = torch.fft.ifftshift(x_padded, dim=(-2, -1))
        x_fft2 = torch.fft.fft2(x_fft2, dim=(-2, -1))
        x_fft2 = torch.fft.fftshift(x_fft2, dim=(-2, -1))
        
        # Polar coordinate transformation
        polar_image = self.polar_transform(x_fft2)
        
        # 1D Fourier transform along the radial direction
        polar_image = torch.fft.ifftshift(polar_image, dim=-1)
        result = torch.fft.ifft(polar_image, dim=-1)
        result = torch.fft.fftshift(result, dim=-1)

        # Crop back to original size only on dim -1 
        # result = result[:, :, :, self.num_col//2:-self.num_col//2]

        return result

    def inverse(self, y):

        # invert the forward step by step
        y = torch.fft.ifftshift(y, dim=-1)
        y = torch.fft.fft(y, dim=-1)
        y = torch.fft.fftshift(y, dim=-1)

        # inverse the polar coordinate transformation
        x_fft2 = self.polar_transform.inverse(y, max_iter=200, verbose=True)

        # inverse the 2D Fourier transform
        x_fft2 = torch.fft.ifftshift(x_fft2, dim=(-2, -1))
        x = torch.fft.ifft2(x_fft2, dim=(-2, -1))
        x = torch.fft.fftshift(x, dim=(-2, -1))

        # crop back to original size
        x = x[:, :, self.num_row//3:-self.num_row//3, self.num_col//3:-self.num_col//3]

        return x
    
    def to(self, device):
        self.polar_transform.to(device)
        return super(CTProjector_FourierSliceTheorem, self).to(device)
    