
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
    
    def forward_LinearOperator(self):
        return self
    
    def adjoint_LinearOperator(self):
        return AdjointLinearOperator(self)    
    
    def _pseudo_inverse_weighted_average(self, x):
        """
        This method implements the pseudo inverse of the linear operator using a weighted average.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to the adjoint of the linear operator.
        returns:
            adj_result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the adjoint of the linear operator to the input tensor.
        """
        
        numerator = self.adjoint(x)
        
        denominator = self.adjoint(torch.ones_like(x))
        
        return numerator / (denominator + 1e-10)  # Avoid division by zero

    def _pseudo_inverse_CG(self, b, max_iter=1000, tol=1e-6, reg_strength=1e-3, verbose=False):
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
        x_est = self._pseudo_inverse_weighted_average(b)
        
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
    
    def pseudo_inverse(self, y, method=None, **kwargs):
        """
        This method implements the pseudo inverse of the linear operator.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to which the pseudo inverse of the linear operator should be applied.
            method: str
                The method to use for computing the pseudo inverse. If None, the method is chosen automatically.
            kwargs: dict
                Keyword arguments to be passed to the method.
        """

        if method is None:
            method = "CG"

        assert method in ["weighted_average", "CG"], "The method should be either 'weighted_average' or 'CG'."

        if method == "weighted_average":
            result =  self._pseudo_inverse_weighted_average(y, **kwargs)
        elif method == "CG":
            result =  self._pseudo_inverse_CG(y, **kwargs)

        return result


class AdjointLinearOperator(LinearOperator):
    def __init__(self, linear_operator: LinearOperator):
        """
        This is an abstract class for linear operators that can be applied to their adjoint.

        It inherits from LinearOperator.

        parameters:
            linear_operator: LinearOperator object
                The linear operator to which the adjoint should be applied.
        """
            
        assert isinstance(linear_operator, LinearOperator), "The linear operator should be a LinearOperator object."

        super(AdjointLinearOperator, self).__init__(linear_operator.output_shape, linear_operator.input_shape)

        self.linear_operator = linear_operator  
        
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

        return self.linear_operator.adjoint(x)
    
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

        return self.linear_operator.forward(y)

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

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the inverse of the linear operator.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to which the inverse linear operator should be applied.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of appl\ng the inverse linear operator to the input tensor.
        """
        raise NotImplementedError
    
    def inverse_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the inverse of the adjoint of the linear operator.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to which the inverse of the adjoint of the linear operator should be applied.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the inverse of the adjoint of the linear operator to the input tensor.
        """
        raise NotImplementedError
    
    def inverse_LinearOperator(self):
        return InverseLinearOperator(self)
    
    def inverse_adjoint_LinearOperator(self):
        return InverseLinearOperator(self.adjoint_LinearOperator())
    
class InverseLinearOperator(LinearOperator):
    def __init__(self, linear_operator: SquareLinearOperator):
        """
        This is an abstract class for linear operators that can be inverted.

        It inherits from LinearOperator.

        parameters:
            linear_operator: SquareLinearOperator object
                The linear operator to which the inverse should be applied.
        """
            
        assert isinstance(linear_operator, SquareLinearOperator), "The linear operator should be a SquareLinearOperator object."

        super(InverseLinearOperator, self).__init__(linear_operator.output_shape, linear_operator.input_shape)

        self.linear_operator = linear_operator  
        
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

        return self.linear_operator.inverse(x)
    
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

        return self.linear_operator.inverse_adjoint(y)
    
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
        return self.linear_operator.forward(y)
    
    def inverse_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the inverse of the adjoint of the linear operator.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to which the inverse of the adjoint of the linear operator should be applied.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the inverse of the adjoint of the linear operator to the input tensor.
        """
        return self.linear_operator.adjoint(y)
    
    def inverse_LinearOperator(self):
        return self.linear_operator
    
    def inverse_adjoint_LinearOperator(self):
        return self.linear_operator.adjoint_LinearOperator()






    

class ScalarLinearOperator(SquareLinearOperator):
    def __init__(self, input_shape, scalar):
        """
        This class implements a scalar linear operator.

        It inherits from SquareLinearOperator.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
            scalar: float
                The scalar to multiply the input tensor with.
        """

        super(ScalarLinearOperator, self).__init__(input_shape)

        self.scalar = scalar

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

        return self.scalar * x
    
    def adjoint(self, y):
        """
        This method implements the adjoint pass of the linear operator, i.e. the conjugate-transposed matrix-vector product.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to the adjoint of the linear operator.
        returns:
            adj_result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the adjoint of the linear operator to the input tensor.
        """

        return self.scalar * y
    
    def inverse(self, y):
        """
        This method implements the inverse of the linear operator.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to which the inverse linear operator should be applied.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the inverse linear operator to the input tensor.
        """

        return y / self.scalar
    
    def inverse_adjoint(self, x):
        """
        This method implements the inverse of the adjoint of the linear operator.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to which the inverse of the adjoint of the linear operator should be applied.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *input_shape
                The result of applying the inverse of the adjoint of the linear operator to the input tensor.
        """
            
        return x / self.scalar  
    
class DiagonalLinearOperator(SquareLinearOperator):
    def __init__(self, input_shape, diagonal):
        """
        This class implements a diagonal linear operator.

        It inherits from SquareLinearOperator.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
            diagonal: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The diagonal of the linear operator.
        """

        super(DiagonalLinearOperator, self).__init__(input_shape)

        self.diagonal = diagonal
    
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
        return self.diagonal * x
    
    def adjoint(self, y):
        """
        This method implements the adjoint pass of the linear operator, i.e. the conjugate-transposed matrix-vector product.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to the adjoint of the linear operator.
        returns:
            adj_result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the adjoint of the linear operator to the input tensor.
        """
        return self.diagonal * y

    def inverse(self, y):
        """
        This method implements the inverse of the linear operator.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to which the inverse linear operator should be applied.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the inverse linear operator to the input tensor.
        """
        return y / self.diagonal
    
    def inverse_adjoint(self, x):
        """
        This method implements the inverse of the adjoint of the linear operator.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to which the inverse of the adjoint of the linear operator should be applied.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *input_shape
                The result of applying the inverse of the adjoint of the linear operator to the input tensor.
        """
        return x / self.diagonal
    



class UnitaryLinearOperator(LinearOperator):
    def __init__(self, input_shape):
        """
        This is an abstract class for unitary linear operators.

        It inherits from InvertibleLinearOperator.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
        """

        super(UnitaryLinearOperator, self).__init__(input_shape, input_shape)

    def inverse(self, y):
        """
        This method implements the inverse of the linear operator.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to which the inverse linear operator should be applied.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the inverse linear operator to the input tensor.
        """

        return self.adjoint(y)


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
    


class IdentityLinearOperator(UnitaryLinearOperator, HermitianLinearOperator):
    def __init__(self, input_shape):
        """
        This class implements the identity linear operator.

        It inherits from SquareLinearOperator.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
        """

        SquareLinearOperator.__init__(self, input_shape)

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

        return x
    
    def inverse_adjoint(self, x):
        """
        This method implements the inverse of the adjoint of the linear operator.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to which the inverse of the adjoint of the linear operator should be applied.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the inverse of the adjoint of the linear operator to the input tensor.
        """
        # this is just to be cute, could just return x. 
        # Identity is the only case where inverse_adjoint is the same as forward
        return self.forward(x) 



class CompositeLinearOperator(LinearOperator):
    def __init__(self, linear_operators):
        """
        This class represents the matrix-matrix product of multiple linear operators.

        It inherits from LinearOperator.

        parameters:
            operators: list of LinearOperator objects
                The list of linear operators to be composed. The product is taken in the order they are provided.
        """

        assert isinstance(linear_operators, list), "The operators should be provided as a list of LinearOperator objects."
        assert len(linear_operators) > 0, "At least one operator should be provided."
        for operator in linear_operators:
            assert isinstance(operator, LinearOperator), "All operators should be LinearOperator objects."

        # The input shape of the composite operator is the input shape of the first operator,
        # and the output shape is the output shape of the last operator.
        input_shape = linear_operators[0].input_shape
        output_shape = linear_operators[-1].output_shape

        LinearOperator.__init__(self, input_shape, output_shape)

        self.linear_operators = linear_operators

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
        for linear_operator in self.linear_operators:
            result = linear_operator.forward(result)
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

        result = y
        # Apply the adjoint of each operator in reverse order
        for linear_operator in reversed(self.linear_operators):
            result = linear_operator.adjoint(result)
        return result
    
    def inverse(self, y):
        """
        This method implements the inverse of the composite linear operator.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to which the inverse of the composite linear operator should be applied.
        returns:
            result: torch.Tensor
                The result of applying the inverse of the composite linear operator to the input tensor.
        """

        result = y
        # Apply the inverse of each operator in reverse order
        for linear_operator in reversed(self.linear_operators):
            assert isinstance(linear_operator, SquareLinearOperator), "The inverse of a composite linear operator can only be computed if all operators are square."
            result = linear_operator.inverse(result)
        return result
    
    def inverse_adjoint(self, y):
        """
        This method implements the inverse of the adjoint of the composite linear operator.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to which the inverse of the adjoint of the composite linear operator should be applied.
        returns:
            result: torch.Tensor
                The result of applying the inverse of the adjoint of the composite linear operator to the input tensor.
        """

        result = y
        # Apply the inverse of the adjoint of each operator in reverse order
        for linear_operator in self.linear_operators:
            assert isinstance(linear_operator, SquareLinearOperator), "The inverse of the adjoint of a composite linear operator can only be computed if all operators are square."
            result = linear_operator.inverse_adjoint(result)

    def to(self, device):
        for i in range(len(self.linear_operators)):
            self.linear_operators[i].to(device)
        return super(CompositeLinearOperator, self).to(device)

class EigenDecomposedLinearOperator(CompositeLinearOperator):
    def __init__(self, input_shape, eigenvalues: DiagonalLinearOperator, eigenvectors: UnitaryLinearOperator):
        """
        This class represents a linear operator that is given by its eigenvalue decomposition.

        It inherits from SquareLinearOperator.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
            eigenvalues: DiagonalLinearOperator object
                The diagonal matrix of eigenvalues.
            eigenvectors: UnitaryLinearOperator object
                The matrix of eigenvectors.
        """

        assert isinstance(eigenvalues, DiagonalLinearOperator), "The eigenvalues should be a DiagonalLinearOperator object."
        assert isinstance(eigenvectors, UnitaryLinearOperator), "The eigenvectors should be a UnitaryLinearOperator object."
        assert eigenvalues.input_shape == input_shape, "The input shape of the eigenvalues should be the same as the input shape of the linear operator."
        assert eigenvectors.input_shape == input_shape, "The input shape of the eigenvectors should be the same as the input shape of the linear operator."
        assert eigenvalues.output_shape == input_shape, "The output shape of the eigenvalues should be the same as the input shape of the linear operator."
        assert eigenvectors.output_shape == input_shape, "The output shape of the eigenvectors should be the same as the input shape of the linear operator."

        operators = [eigenvectors, eigenvalues, eigenvectors.adjoint_LinearOperator()]

        super(EigenDecomposedLinearOperator, self).__init__(operators)

        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

class SingularValueDecomposedLinearOperator(CompositeLinearOperator):
    def __init__(self, input_shape, left_singular_vectors: UnitaryLinearOperator, singular_values: DiagonalLinearOperator,  right_singular_vectors: UnitaryLinearOperator):
        """
        This class represents a linear operator that is given by its singular value decomposition.

        It inherits from SquareLinearOperator.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
            singular_values: DiagonalLinearOperator object
                The diagonal matrix of singular values.
            left_singular_vectors: UnitaryLinearOperator object
                The matrix of left singular vectors.
            right_singular_vectors: UnitaryLinearOperator object
                The matrix of right singular vectors.
        """

        assert isinstance(singular_values, DiagonalLinearOperator), "The singular values should be a DiagonalLinearOperator object."
        assert isinstance(left_singular_vectors, UnitaryLinearOperator), "The left singular vectors should be a UnitaryLinearOperator object."
        assert isinstance(right_singular_vectors, UnitaryLinearOperator), "The right singular vectors should be a UnitaryLinearOperator object."
        assert singular_values.input_shape == input_shape, "The input shape of the singular values should be the same as the input shape of the linear operator."
        assert left_singular_vectors.input_shape == input_shape, "The input shape of the left singular vectors should be the same as the input shape of the linear operator."
        assert right_singular_vectors.input_shape == input_shape, "The input shape of the right singular vectors should be the same as the input shape of the linear operator."
        assert singular_values.output_shape == input_shape, "The output shape of the singular values should be the same as the input shape of the linear operator."
        assert left_singular_vectors.output_shape == input_shape, "The output shape of the left singular vectors should be the same as the input shape of the linear operator."
        assert right_singular_vectors.output_shape == input_shape, "The output shape of the right singular vectors should be the same as the input shape of the linear operator."

        operators = [left_singular_vectors, singular_values, right_singular_vectors.adjoint_LinearOperator()]

        super(SingularValueDecomposedLinearOperator, self).__init__(operators)

        self.singular_values = singular_values
        self.left_singular_vectors = left_singular_vectors
        self.right_singular_vectors = right_singular_vectors