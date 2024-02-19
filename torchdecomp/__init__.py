"""A set of matrix and tensor decomposition models
   implemented as PyTorch classes
"""
from .helper import is_symmetric_matrix, create_dummy_matrix
from .helper import print_named_parameters
from .lu import LULayer
from .cholesky import CholeskyLayer
from .qr import QRLayer
from .factor import FactorLayer
from .rec import RecLayer
from .symrec import SymRecLayer
from .ica import RotationLayer, KurtosisICALayer
from .ica import NegentropyICALayer, DDICALayer
from .nmf import NMFLayer


# Object Export
__all__ = [
    # Helper functions
    "is_symmetric_matrix", "create_dummy_matrix", "print_named_parameters",
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
    "NMFLayer"
]
