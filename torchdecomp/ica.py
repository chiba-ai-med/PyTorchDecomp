import torch
import torch.nn as nn
import math


# Rotation Matrix
class RotationLayer(nn.Module):
    """Rotation Matrix Factorization Layer

    A symmetic matrix X (n times n) is rotated by
    a rotation matrix A such as XA (n times n).

    Attributes:
        size (int): The number of rows of X (n)

    Example:
        >>> import torchdecomp as td
        >>> import torch
        >>> torch.manual_seed(123456)
        >>> x = torch.randn(6, 6) # Test datasets
        >>> rotation_layer = td.RotationLayer(x.size(0)) # Instantiation

    """
    def __init__(self, size):
        """Initialization function
        """
        super(RotationLayer, self).__init__()
        self.mixing_matrix = nn.Parameter(
            torch.randn(size, size))
    
    def forward(self, x):
        """Forward propagation function
        """
        x = torch.tanh(x @ self.mixing_matrix)
        return x


# Kurtosis-based Independent Component Analysis
class KurtosisICALayer(nn.Module):
    """Kurtosis-based Independent Component Analysis Layer

    Mini-batch data (x) is used.

    Example:
        >>> import torchdecomp as td
        >>> import torch
        >>> torch.manual_seed(123456)
        >>> x = torch.randn(6, 6) # Test datasets
        >>> rotation_layer = td.RotationLayer(x.size(0)) # Instantiation
        >>> loss = td.KurtosisICALayer()
        >>> loss(rotation_layer)

    """
    def __init__(self):
        """Initialization function
        """
        super(KurtosisICALayer, self).__init__()
    
    def forward(self, x):
        """Forward propagation function
        """
        numer = ((x - x.mean(dim=0))**4).mean(dim=0)
        denom = ((x - x.mean(dim=0))**2).mean(dim=0)**2
        kurtosis = numer / denom
        loss = - torch.sum(kurtosis)
        return loss


# Negentropy-based Independent Component Analysis
class NegentropyICALayer(nn.Module):
    """Negentropy-based Independent Component Analysis Layer

    Mini-batch data (x) is used.

    Example:
        >>> import torchdecomp as td
        >>> import torch
        >>> torch.manual_seed(123456)
        >>> x = torch.randn(10, 6) # Test datasets
        >>> negentropy_ica_layer = td.NegentropyICALayer(x) # Instantiation

    """
    def __init__(self):
        """Initialization function
        """
        super(NegentropyICALayer, self).__init__()
    
    def forward(self, x):
        """Forward propagation function
        """
        numer = - (x - x.mean(dim=0))**2
        denom1 = 2 * (x.std(dim=0) + 1e-8)**2
        denom2 = (2 * math.pi)**0.5 * (x.std(dim=0) + 1e-8)
        negentropy = - (torch.exp(numer / denom1) / denom2).log()
        loss = - torch.sum(negentropy)
        return loss


# Deep Deterministic Independent Component Analysis
class _GramMatrixLayer(nn.Module):
    def __init__(self, sigma):
        """Initialization function
        """
        super(_GramMatrixLayer, self).__init__()
        self.sigma = sigma
    
    def forward(self, x):
        """Forward propagation function
        """
        G = torch.cdist(x, x)
        G = torch.exp(-G.pow(2) / (2 * self.sigma**2))
        A = G / torch.trace(G)
        return A


class _EigenValsLayer(nn.Module):
    def __init__(self):
        """Initialization function
        """
        super(_EigenValsLayer, self).__init__()
    
    def forward(self, x):
        """Forward propagation function
        """
        return torch.linalg.eigvals(x)


class _EntropyLayer(nn.Module):
    def __init__(self, alpha):
        """Initialization function
        """
        super(_EntropyLayer, self).__init__()
        self.alpha = alpha
    
    def forward(self, x):
        """Forward propagation function
        """
        x = x ** self.alpha
        x = torch.sum(x)
        x = torch.log2(x)
        x = x / (1 - self.alpha)
        return x


class _HadamardProdLayer(nn.Module):
    def __init__(self):
        """Initialization function
        """
        super(_HadamardProdLayer, self).__init__()
    
    def forward(self, x):
        """Forward propagation function
        """
        return x * x


class DDICALayer(nn.Module):
    """Deep Deterministic Independent Component Analysis-based
    Independent Component Analysis Layer

    Mini-batch data (x) is used.

    Example:
        >>> import torchdecomp as td
        >>> import torch
        >>> torch.manual_seed(123456)
        >>> x = torch.randn(6, 6) # Test datasets
        >>> rotation_layer = td.DDICALayer(x.size(0)) # Instantiation

    Note:
       This model is very initial-value sensitive.
       If the iteration is not proceeded, re-run sometimes.

    """
    def __init__(self, sigma, alpha, size):
        """Initialization function
        """
        super(DDICALayer, self).__init__()
        self.mixing_matrix = nn.Parameter(
            torch.rand(size, size, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.size = nn.Parameter(
            torch.tensor(size, dtype=torch.float32))
        self.seq_each = nn.Sequential(
            _GramMatrixLayer(self.sigma),
            _EigenValsLayer(),
            _EntropyLayer(self.alpha))
        
        self.seq_joint = nn.Sequential(*[nn.Sequential(
            _GramMatrixLayer(self.sigma)) for _ in range(size)],
            _HadamardProdLayer(),
            _EigenValsLayer(),
            _EntropyLayer(self.alpha))
    
    def forward(self, x):
        """Forward propagation function
        """
        x = torch.mm(x, self.mixing_matrix)
        return torch.real(self.seq_each(x) - self.seq_joint(x))
