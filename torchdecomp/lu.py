import torch
import torch.nn as nn


class LULayer(nn.Module):
    """LU Decomposition Layer

    An square matrix X (n times n) is decomposed to
    the product of L (n times n) and U (n times n).

    Attributes:
        size (int): The size of an square matrix (n)

    Example:
        >>> import torchdecomp as td
        >>> import torch
        >>> torch.manual_seed(123456)
        >>> x = torch.randn(6, 6) # Test datasets
        >>> lu_layer = td.LULayer(x.size()) # Instantiation

    """
    def __init__(self, size):
        """Initialization function
        """
        super(LULayer, self).__init__()
        if size[0] != size[1]:
            raise ValueError("LULayer supports square matrices only.")
        L = torch.tril(torch.randn(size), diagonal=-1)
        U = torch.triu(torch.randn(size), diagonal=1)
        # Set diagonal elements as 1s
        for i in range(min(size)):
            L[i, i] = 1.0
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(U)
    
    def forward(self):
        """Forward propagation function
        """
        return torch.mm(self.L, self.U)
