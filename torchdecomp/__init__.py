"""A set of matrix decomposition algorithms
implemented as PyTorch classes
"""
from .helper import create_dummy_matrix, print_named_parameters
from .lu import LULayer
from .cholesky import CholeskyLayer
from .qr import QRLayer
from .factor import FactorLayer
from .rec import RecLayer
from .symrec import SymRecLayer
from .ica import RotationLayer, KurtosisICALayer
from .ica import NegentropyICALayer, DDICALayer
from .nmf import NMFLayer, gradNMF, updateNMF


# Object Export
__all__ = [
    # Helper functions
    "create_dummy_matrix", "print_named_parameters",
    # LU Decomposition
    "LULayer",
    # Cholesky Decomposition
    "CholeskyLayer",
    # QR Decomposition
    "QRLayer",
    # Factor Matrix
    "FactorLayer",
    # Reconstruction Matrix
    "RecLayer",
    # Symmetric Reconstruction Matrix
    "SymRecLayer",
    # Independent Component Analysis
    "RotationLayer", "KurtosisICALayer", "NegentropyICALayer", "DDICALayer",
    # Non-negative Matrix Factorization
    "NMFLayer", "gradNMF", "updateNMF"
]
