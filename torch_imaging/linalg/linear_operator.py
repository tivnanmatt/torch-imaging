import torch

class LinearOperator(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        """
        This is an abstract class for linear operators.

        It inherits from torch.nn.Module.
        
        It requires the methods forward and adjoint to be implemented.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
            output_shape: tuple of integers
                The shape of the output tensor, disregarding batch and channel dimensions.
        """

        super(LinearOperator, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method implements the forward pass of the linear operator, i.e. the matrix-vector product.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to the linear operator.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The result of applying the linear operator to the input tensor.
        """

        raise NotImplementedError
    
    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the adjoint pass of the linear operator, i.e. the conjugate-transposed matrix-vector product.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to the adjoint of the linear operator.
        returns:
            adj_result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the adjoint of the linear operator to the input tensor.
        """

        raise NotImplementedError
    
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
                print("Inverting ", self.__class__.__name__, " with CG. Iteration: {}, Residual: {}".format(i, torch.sqrt(torch.abs(rsold))))
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
    

class SquareLinearOperator(LinearOperator):
    def __init__(self, input_shape):
        """
        This is an abstract class for square linear operators.

        It inherits from LinearOperator.

        For square linear operators, the input and output shapes are the same.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
        """

        super(SquareLinearOperator, self).__init__(input_shape, input_shape)


class HermitianLinearOperator(SquareLinearOperator):
    def __init__(self, input_shape):
        """
        This is an abstract class for Hermitian, or self-adjoint linear operators.

        It inherits from SquareLinearOperator.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
        """

        super(HermitianLinearOperator, self).__init__(input_shape)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the adjoint pass of the linear operator, i.e. the matrix-vector product with the adjoint.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to the adjoint of the linear operator.
        returns:
            adj_result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the adjoint of the linear operator to the input tensor.
        """

        return self.forward(y)
    

class InvertibleLinearOperator(SquareLinearOperator):
    def __init__(self, input_shape):
        """
        This is an abstract class for invertible linear operators.

        Modules which check for this label will assume the class implements the inverse method

        It inherits from SquareLinearOperator.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
        """

        super(InvertibleLinearOperator, self).__init__(input_shape)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the inverse of the linear operator.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to which the inverse linear operator should be applied.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the inverse linear operator to the input tensor.
        """

        raise NotImplementedError
    

class PseudoInvertibleLinearOperator(LinearOperator):
    def __init__(self, input_shape):
        """
        This is an abstract class for pseudo-invertible linear operators.

        Modules which check for this label will assume the class implements the pseudo_inverse method

        It inherits from LinearOperator.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
        """

        super(PseudoInvertibleLinearOperator, self).__init__(input_shape)

    def pseudo_inverse(self, y):
        """
        This method implements the pseudo-inverse of the linear operator.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to which the pseudo-inverse linear operator should be applied.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the pseudo-inverse linear operator to the input tensor.
        """

        raise NotImplementedError
    




class CompositeLinearOperator(LinearOperator):
    def __init__(self, operators):
        """
        This class represents the matrix-matrix product of multiple linear operators.

        It inherits from LinearOperator.

        parameters:
            operators: list of LinearOperator objects
                The list of linear operators to be composed. The product is taken in the order they are provided.
        """

        assert isinstance(operators, list), "The operators should be provided as a list."
        assert len(operators) > 0, "At least one operator should be provided."
        for operator in operators:
            assert isinstance(operator, LinearOperator), "All operators should be LinearOperator objects."

        # The input shape of the composite operator is the input shape of the first operator,
        # and the output shape is the output shape of the last operator.
        input_shape = operators[0].input_shape
        output_shape = operators[-1].output_shape

        super(CompositeLinearOperator, self).__init__(input_shape, output_shape)

        self.operators = operators

    def forward(self, x):
        """
        This method implements the forward pass of the composite linear operator.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to the composite linear operator.
        returns:
            result: torch.Tensor
                The result of applying the composite linear operator to the input tensor.
        """

        result = x
        for operator in self.operators:
            result = operator(result)
        return result
    
    def adjoint(self, y):
        """
        This method implements the adjoint pass of the composite linear operator.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to the adjoint of the composite linear operator.
        returns:
            adj_result: torch.Tensor
                The result of applying the adjoint of the composite linear operator to the input tensor.
        """

        adj_result = y
        # Apply the adjoint of each operator in reverse order
        for operator in reversed(self.operators):
            adj_result = operator.adjoint(adj_result)
        return adj_result
    


class CompositeInvertibleLinearOperator(CompositeLinearOperator, InvertibleLinearOperator):
    def __init__(self, operators):
        """
        Represents a composite linear operator where all constituent operators are invertible.

        parameters:
            operators: list of InvertibleLinearOperator objects
                The list of linear operators to be composed. All must be invertible.
        """

        # Check that all operators are invertible
        for operator in operators:
            assert isinstance(operator, InvertibleLinearOperator), "All operators must be invertible."

        super(CompositeInvertibleLinearOperator, self).__init__(operators)

    def inverse(self, y):
        """
        Implements the inverse of the composite linear operator.

        parameters:
            y: torch.Tensor
                The input tensor to which the inverse linear operator should be applied.
        returns:
            result: torch.Tensor
                The result of applying the inverse linear operator to the input tensor.
        """

        result = y
        # Apply the inverse of each operator in reverse order
        for operator in reversed(self.operators):
            result = operator.inverse(result)
        return result