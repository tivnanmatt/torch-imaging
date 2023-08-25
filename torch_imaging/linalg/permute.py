
import torch

from .linear_operator import LinearOperator

class Pad(LinearOperator):
    def __init__(self, input_shape, pad_width, mode='constant', value=0):
        """
        This class implements a padding operator that can be used in a PyTorch model.

        it assumes the central pixel in the image is at (0,0)

        it returns the padded input
        """
        super(Pad, self).__init__(input_shape)
        self.pad_width = pad_width
        self.mode = mode
        self.value = value

    def forward(self, x):
        return torch.nn.functional.pad(x, self.pad_width, mode=self.mode, value=self.value)

    def adjoint(self, y):
        return torch.nn.functional.pad(y, [-self.pad_width[i] for i in range(len(self.pad_width))], mode=self.mode, value=self.value)
    
    def pseudo_inverse(self, y, **kwargs):
        # A^T A is identity, so (A^T A)^-1 A^T is just A^T
        return self.adjoint(y)