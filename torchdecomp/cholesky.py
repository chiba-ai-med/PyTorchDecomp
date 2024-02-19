import torch
import torch.nn as nn


class CholeskyLayer(nn.Module):
    """Cholesky Decomposition Layer

    A symmetric matrix X (n times n) is decomposed to
    the product of L (n times n) and L^T (n times n).

    Attributes:
        size (int): The size of a symmetric matrix (n)

    Example:
        >>> import torchdecomp as td
        >>> import torch
        >>> torch.manual_seed(123456)
        >>> x = torch.randn(6, 6) # Test datasets
        >>> x = torch.mm(x, x.t()) # Symmetalization
        >>> cholesky_layer = td.CholeskyLayer(x.size()) # Instantiation

    """
    def __init__(self, size):
        """Initialization function
        """
        super(CholeskyLayer, self).__init__()
        if size[0] != size[1]:
            raise ValueError("CholeskyLayer supports square matrices only.")
        L = torch.tril(torch.randn(size))
        # Set diagonal elements as positive values
        for i in range(min(size)):
            L[i, i] = torch.exp(L[i, i])
        self.L = nn.Parameter(L)
    
    def forward(self):
        """Forward propagation function
        """
        return torch.mm(self.L, self.L.t())
