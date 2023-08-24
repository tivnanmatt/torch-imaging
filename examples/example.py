



   

# Testing the PolarCoordinateTransformation module
theta = torch.linspace(0, 2 * pi, 360)
radius = torch.linspace(-200, 200, 1000)

cart2pol = PolarCoordinateTransformation(256, 256, theta, radius)

# now make a (72, 72) x,y coordinate meshgrid
x_mesh, y_mesh = torch.meshgrid(torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256))

# define a spiral
theta = torch.linspace(0, 10 * pi, 1000)
radius = torch.linspace(0, 1, 1000)

# convert to x,y coordinates
x = radius * torch.cos(theta)
y = radius * torch.sin(theta)

# make a mask of those x,y coordinates
mask = torch.zeros_like(x_mesh, dtype=torch.float32, device=device)
for i in range(len(x)):
    mask[(x_mesh - x[i])**2 + (y_mesh - y[i])**2 < .001] = 1

# show the mask
from matplotlib import pyplot as plt
plt.imshow(mask.cpu())
plt.show()

# reshape it to a (1, 1, 72, 72) tensor
mask = mask.view(1, 1, 256, 256)
mask = mask.type(torch.float32)
# mask = mask*0 + 1

# apply the polar coordinate transformation
output = cart2pol(mask)
# output = output + torch.randn_like(output)*0.5
# show the output
plt.imshow(output[0, 0].cpu())
plt.show()

# Applying the inverse transformation
inverse_output = cart2pol.inverse(output)
# inverse_output = cart2pol.inverse(output*0 + 1)

# Show the output of the inverse transformation
plt.imshow(inverse_output[0, 0].cpu(),vmin=0,vmax=1)
plt.show(block=True)




from matplotlib import pyplot as plt


# Testing the FourierSliceForwardProjector module

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


forward_projector = CTProjector_FourierSliceTheorem(3*image_size, 3*image_size, theta, radius)

# Create a (1, 1, 256, 256) tensor
x = torch.zeros(1, 1, 256, 256).to(device)

# Set the center pixel to 1
# x[0, 0, 100:120, 100:120] = 1
x = mask.clone()

# Apply the forward projector
y = forward_projector(x)

# Show the result
plt.figure()
plt.imshow(y[0, 0].cpu().abs(), aspect='auto')
plt.show(block=True)

print()



x_inv = forward_projector.inverse(y + 10*torch.randn_like(y))

plt.figure()
plt.imshow(x_inv[0, 0].cpu().real, aspect='auto')
plt.show(block=True)