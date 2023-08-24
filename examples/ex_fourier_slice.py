
import torch
from matplotlib import pyplot as plt

# define pi. We avoid using numpy to avoid conflicts with pytorch
pi = 3.1415927410125732

# get the path to the directory containing this file and add the previous directory to the path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# now we can import from torch_imaging
from torch_imaging.linalg.ct_projector import CTProjector_FourierSliceTheorem


# if the GPU is available, use it
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Testing the FourierSliceForwardProjector module

# now make a (256, 256) x,y coordinate meshgrid
x_mesh, y_mesh = torch.meshgrid(torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256))

# define a spiral
theta = torch.linspace(0, 10 * pi, 1000)
radius = torch.linspace(0, 1, 1000)

# convert to x,y coordinates
x = radius * torch.cos(theta)
y = radius * torch.sin(theta)

# make a mask of those x,y coordinates
spiral = torch.zeros_like(x_mesh, dtype=torch.float32, device=device)
for i in range(len(x)):
    spiral[(x_mesh - x[i])**2 + (y_mesh - y[i])**2 < .001] = 1

# show the mask
plt.imshow(spiral.cpu())
plt.title('Spiral Mask')
plt.show()

image_size = 256
num_fourier_radial_samples = 1024
num_fourier_angular_samples = 1440

# Create a FourierSliceForwardProjector module
theta = torch.linspace(0, 2 * pi, num_fourier_angular_samples)
num_samples = 1024
sample_spacing = 1.5*image_size*3/num_fourier_radial_samples # we use 3x zero padding so this is the spacing in resulting fourier space

# Create two separate tensors for the negative and positive ranges
negative_samples = torch.arange(start=-sample_spacing * (num_samples//2), end=-sample_spacing, step=sample_spacing)
positive_samples = torch.arange(start=0, end=sample_spacing * (num_samples//2), step=sample_spacing)

# Concatenate the tensors to get the final radius tensor
radius = torch.cat([negative_samples, positive_samples], dim=0)

# we are using an even number of samples, 
forward_projector = CTProjector_FourierSliceTheorem(3*image_size, 3*image_size, theta, radius).to(device)

# Apply the forward projector
y = forward_projector(spiral.unsqueeze(0).unsqueeze(0))

# Show the result
plt.figure()
plt.imshow(y[0, 0].cpu().abs(), aspect='auto')
plt.title('Forward Projection of Spiral Mask')
plt.show(block=True)

# Apply the inverse projector
x_inv = forward_projector.inverse(y)

# Show the result
plt.figure()
plt.imshow(x_inv[0, 0].cpu().real, aspect='auto')
plt.title('Reconstructed Spiral Mask')
plt.show(block=True)