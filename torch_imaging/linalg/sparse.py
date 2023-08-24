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

        return result

    def adjoint(self, x):
        """
        This method implements the adjoint pass of the linear operator, i.e. the matrix-vector product with the adjoint.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to the adjoint of the linear operator.
        returns:
            adj_result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the adjoint of the linear operator to the input tensor.
        """

        batch_size, num_channel = x.shape[:2]
        
        assert x.shape[2:] == self.output_shape, "Input tensor shape doesn't match the specified output shape."
        
        adj_result = torch.zeros(batch_size, num_channel, *self.input_shape, dtype=x.dtype, device=x.device)
        
        # Flatten the adjoint result tensor
        adj_result_flattened = adj_result.view(batch_size, num_channel, -1)
        x_flattened = x.view(batch_size, num_channel, -1)

        for i in range(self.indices.shape[0]):
            for b in range(batch_size):
                for c in range(num_channel):
                    adj_result_flattened[b, c].index_add_(0, self.indices[i], (x_flattened[b, c] * self.weights[i]))

        return adj_result

    def pseudo_inverse_weighted_average(self, x):
        """
        This method implements the pseudo inverse of the linear operator using a weighted average.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to the adjoint of the linear operator.
        returns:
            adj_result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the adjoint of the linear operator to the input tensor.
        """

        batch_size, num_channel, _ = x.shape
        
        numerator = self.adjoint(x)
        
        ones_tensor = torch.ones_like(x)
        denominator = self.adjoint(ones_tensor)
        
        return numerator / (denominator + 1e-10)  # Avoid division by zero

    def pseudo_inverse_CG(self, b, max_iter=1000, tol=1e-6, reg_strength=1e-3, verbose=False):
        """
        This method implements the pseudo inverse of the linear operator using the conjugate gradient method.

        It solves the linear system (A^T A + reg_strength * I) x = A^T b for x, where A is the linear operator.

        parameters:
            b: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to which the pseudo inverse of the linear operator should be applied.
            max_iter: int
                The maximum number of iterations to run the conjugate gradient method.
            tol: float
                The tolerance for the conjugate gradient method.
            reg_strength: float
                The regularization strength for the conjugate gradient method.
        returns:
            x_est: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the adjoint of the linear operator to the input tensor.
        """
        ATb = self.adjoint(b)
        x_est = self.pseudo_inverse_weighted_average(b)
        
        r = ATb - self.adjoint(self.forward(x_est)) - reg_strength * x_est
        p = r.clone()
        rsold = torch.dot(r.flatten(), r.flatten())
        
        for i in range(max_iter):
            if verbose:
                print("Iteration: {}, Residual: {}".format(i, torch.sqrt(torch.abs(rsold))))
            ATAp = self.adjoint(self.forward(p)) + reg_strength * p
            alpha = rsold / torch.dot(p.flatten(), ATAp.flatten())
            x_est += alpha * p
            r -= alpha * ATAp
            rsnew = torch.dot(r.flatten(), r.flatten())
            if torch.sqrt(torch.abs(rsnew)) < tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
            
        return x_est
    def to(self, *args, **kwargs):
        self.indices = self.indices.to(*args, **kwargs)
        self.weights = self.weights.to(*args, **kwargs)
        return super().to(*args, **kwargs)

