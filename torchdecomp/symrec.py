import torch
import torch.nn as nn


class SymRecLayer(nn.Module):
    """Symmetric Reconstruction Layer

    A symmetric matrix X (n times n) is decomposed to
    the product of U (n times k), Sigma (k times k),
    and U^T (k times n).

    Attributes:
        size (int): The size of a symmetric matrix (n)
        n_components (int): The number of lower dimensions (k)

    Example:
        >>> import torchdecomp as td
        >>> import torch
        >>> torch.manual_seed(123456)
        >>> x = torch.randn(6, 6) # Test datasets
        >>> x = torch.mm(x, x.t()) # Symmetalization
        >>> symrec_layer = td.SymRecLayer(x.size(0), 3) # Instantiation

    """
    def __init__(self, size, n_components):
        """Initialization function
        """
        super(SymRecLayer, self).__init__()
        U = torch.nn.init.orthogonal_(torch.randn(size, n_components), gain=1)
        Sigma = torch.diag(torch.sort(
            torch.randn(n_components)**2, descending=True).values)
        self.U = nn.Parameter(U)
        self.Sigma = nn.Parameter(Sigma)
    
    def forward(self):
        """Forward propagation function
        """
        return torch.mm(torch.mm(self.U, self.Sigma), self.U.t())
