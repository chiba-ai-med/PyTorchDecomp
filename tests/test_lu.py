import torchdecomp as td
import torch
import pytest
import numpy as np


def test_LULayer():
    x = torch.randn(6, 6)
    lu_layer = td.LULayer(x)
    assert lu_layer.L.size()[0] == 6
    assert lu_layer.L.size()[1] == 6
    assert lu_layer.U.size()[0] == 6
    assert lu_layer.U.size()[1] == 6


def test_LULayer_error():
    x = torch.randn(10, 6)
    with pytest.raises(AssertionError) as exc_info:
        td.LULayer(x)
    assert exc_info.type == AssertionError


def test_LULayer_error2():
    x = np.random.rand(6, 6)
    with pytest.raises(AssertionError) as exc_info:
        td.LULayer(x)
    assert exc_info.type == AssertionError
