import torch

from .linear_operator import LinearOperator
    
class RealPart(LinearOperator):
    def __init__(self, input_shape):
        """
        This class converts a complex tensor to a real tensor by taking the real part

        This is not truly a linear operator, but it can be used to convert a real-valued complex64 tensor to a float32 tensor

        """
        super(RealPart, self).__init__(input_shape, input_shape)

    def forward(self, x):
        return x.real
    
    def adjoint(self, y):
        return y.real