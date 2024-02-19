import torch
import torch.nn as nn


class QRLayer(nn.Module):
    """QR Decomposition Layer

    A square matrix X (n times n) is decomposed to
    the product of Q (n times n) and R (m times n).

    Attributes:
        size (int): The size of a symmetric matrix (n)

    Example:
        >>> import torchdecomp as td
        >>> import torch
        >>> torch.manual_seed(123456)
        >>> x = torch.randn(6, 6) # Test datasets
        >>> qr_layer = td.QRLayer(x.size()) # Instantiation

    """
    def __init__(self, size):
        """Initialization function
        """
        super(QRLayer, self).__init__()
        if size[0] != size[1]:
            raise ValueError("QRLayer supports square matrices only.")
        Q = torch.nn.init.orthogonal_(torch.randn(size), gain=1)
        R = torch.triu(torch.randn(size), diagonal=0)
        self.Q = nn.Parameter(Q)
        self.R = nn.Parameter(R)
    
    def forward(self):
        """Forward propagation function
        """
        return torch.mm(self.Q, self.R)
