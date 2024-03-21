"""A set of matrix and tensor decomposition models
   implemented as PyTorch classes
"""
from .helper import create_dummy_matrix, print_named_parameters, rho
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
    "create_dummy_matrix", "print_named_parameters", "rho",
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
