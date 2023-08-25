import torch
import torch.nn as nn

from .linear_operator import LinearOperator

from matplotlib import pyplot as plt

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

pi = 3.1415927410125732



class SparseLinearOperator(LinearOperator):
    def __init__(self, input_shape, output_shape, indices, weights):
        """
        This class implements a sparse linear operator that can be used in a PyTorch model.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
            output_shape: tuple of integers
                The shape of the output tensor, disregarding batch and channel dimensions.
            indices: torch.Tensor of shape [num_weights, *output_shape]
                The 1D indices of the flattened input tensor that each weight corresponds to.
            weights: torch.Tensor of shape [num_weights, *output_shape]
                The weights of the linear operator.
        """

        super(SparseLinearOperator, self).__init__(input_shape, output_shape)

        # Check that indices and weights have the same shape.
        assert indices.shape == weights.shape, "Indices and weights must have the same shape."
        
        self.indices = indices
        self.weights = weights

    def forward(self, x):
        """
        This method implements the forward pass of the linear operator, i.e. the matrix-vector product.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to the linear operator.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The result of applying the linear operator to the input tensor.
        """

        batch_size, num_channel = x.shape[:2]
        
        assert x.shape[2:] == self.input_shape, "Input tensor shape doesn't match the specified input shape."

        result = torch.zeros(batch_size, num_channel, *self.output_shape, dtype=x.dtype, device=x.device)
        
        # Flatten the input tensor
        x_flattened = x.view(batch_size, num_channel, -1)
        
        for i in range(self.indices.shape[0]):  # Loop over num_weights
            values = x_flattened[:, :, self.indices[i]]  # Adding an additional dimension for broadcasting
            result += self.weights[i].view(1,1,-1) * values

        result = result.view(batch_size, num_channel, *self.output_shape)

        return result

    def adjoint(self, x):
        """
        This method implements the adjoint pass of the linear operator, i.e. the matrix-vector product with the adjoint.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to the adjoint of the linear operator.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the adjoint of the linear operator to the input tensor.
        """

        batch_size, num_channel = x.shape[:2]
        
        assert x.shape[2:] == self.output_shape, "Input tensor shape doesn't match the specified output shape."
        
        result = torch.zeros(batch_size, num_channel, *self.input_shape, dtype=x.dtype, device=x.device)
        
        # Flatten the adjoint result tensor
        result_flattened = result.view(batch_size, num_channel, -1)
        x_flattened = x.view(batch_size, num_channel, -1)

        for i in range(self.indices.shape[0]):
            for b in range(batch_size):
                for c in range(num_channel):
                    result_flattened[b, c].index_add_(0, self.indices[i], (x_flattened[b, c] * self.weights[i]))

        return result

    def to(self, *args, **kwargs):
        self.indices = self.indices.to(*args, **kwargs)
        self.weights = self.weights.to(*args, **kwargs)
        return super().to(*args, **kwargs)

