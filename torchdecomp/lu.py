import torch
import torch.nn as nn
from .helper import _check_torch_tensor, _check_square_matrix


class LULayer(nn.Module):
    """LU Decomposition Layer

    A square matrix X (n times n) is decomposed to
    the product of L (n times n) and U (n times n).

    Attributes:
        x (torch.Tensor): A square matrix X (n times n)

    Example:
        >>> import torchdecomp as td
        >>> import torch
        >>> torch.manual_seed(123456)
        >>> x = torch.randn(6, 6) # Test datasets
        >>> lu_layer = td.LULayer(x) # Instantiation

    """
    def __init__(self, x):
        """Initialization function
        """
        super(LULayer, self).__init__()
        _check_torch_tensor(x)
        _check_square_matrix(x)
        size = x.size()
        L = torch.tril(torch.randn(size), diagonal=-1)
        U = torch.triu(torch.randn(size), diagonal=1)
        # Set diagonal elements as 1s
        for i in range(size[0]):
            L[i, i] = 1.0
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(U)

    def forward(self):
        """Forward propagation function
        """
        return torch.mm(self.L, self.U)
