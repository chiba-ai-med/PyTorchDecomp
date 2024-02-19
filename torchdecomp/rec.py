import torch
import torch.nn as nn


class RecLayer(nn.Module):
    """Reconstruction Matrix Layer

    A matrix X (n times m) is projected to
    a smaller matrix XV,
    and then reconstructed such as XVV^T,
    where the size of V is n times k (k << m).

    Attributes:
        size (int): The number of rows of X (n)
        n_components (int): The number of lower dimensions (k)

    Example:
        >>> import torchdecomp as td
        >>> import torch
        >>> torch.manual_seed(123456)
        >>> x = torch.randn(10, 6) # Test datasets
        >>> rec_layer = td.RecLayer(x.size(1), 3) # Instantiation

    """
    def __init__(self, size, n_components):
        """Initialization function
        """
        super(RecLayer, self).__init__()
        V = torch.nn.init.orthogonal_(torch.randn(size, n_components), gain=1)
        self.V = nn.Parameter(V)
    
    def forward(self, x):
        """Forward propagation function
        """
        return torch.mm(torch.mm(x, self.V), self.V.t())
