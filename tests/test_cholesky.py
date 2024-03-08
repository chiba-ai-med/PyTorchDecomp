import torchdecomp as td
import torch
import pytest
import numpy as np


def test_CholeskyLayer():
    x = torch.tensor([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    cholesky_layer = td.CholeskyLayer(x)
    assert cholesky_layer.L.size()[0] == 3
    assert cholesky_layer.L.size()[1] == 3


def test_CholeskyLayer_error():
    x = torch.randn(6, 6)
    with pytest.raises(AssertionError) as exc_info:
        td.CholeskyLayer(x)
    assert exc_info.type == AssertionError


def test_CholeskyLayer_error2():
    x = np.random.rand(6, 6)
    x = np.matmul(x, x.T)
    with pytest.raises(AssertionError) as exc_info:
        td.CholeskyLayer(x)
    assert exc_info.type == AssertionError
