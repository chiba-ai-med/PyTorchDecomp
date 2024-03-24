import torch
import torch.nn as nn
from .helper import _check_torch_tensor, _check_symmetric_matrix
from .helper import _check_dimension


class SymRecLayer(nn.Module):
    """Symmetric Reconstruction Layer

    A symmetric matrix X (n times n) is decomposed to
    the product of Q (n times k), Lambda (k times k),
    and Q^T (k times n).

    Attributes:
        x (torch.Tensor): A symmetric matrix X (n times n)
        n_components (int): The number of lower dimensions (k)

    Example:
        >>> import torchdecomp as td
        >>> import torch
        >>> torch.manual_seed(123456)
        >>> x = torch.randn(6, 6) # Test datasets
        >>> x = torch.mm(x, x.t()) # Symmetalization
        >>> symrec_layer = td.SymRecLayer(x, 3) # Instantiation

    """
    def __init__(self, x, n_components):
        """Initialization function
        """
        super(SymRecLayer, self).__init__()
        _check_torch_tensor(x)
        _check_symmetric_matrix(x)
        size = x.size(0)
        _check_dimension(size, n_components)
        Q = torch.nn.init.orthogonal_(torch.randn(size, n_components), gain=1)
        Lambda = torch.diag(torch.sort(
            torch.randn(n_components)**2, descending=True).values)
        self.Q = nn.Parameter(Q)
        self.Lambda = nn.Parameter(Lambda)

    def forward(self):
        """Forward propagation function
        """
        return torch.mm(torch.mm(self.Q, self.Lambda), self.Q.t())
