import torchdecomp as td
import torch
import pytest


def test_LULayer():
    x = torch.randn(6, 6)
    lu_layer = td.LULayer(x.size())
    assert lu_layer.L.size()[0] == 6
    assert lu_layer.L.size()[1] == 6
    assert lu_layer.U.size()[0] == 6
    assert lu_layer.U.size()[1] == 6


def test_LULayer_error():
    x = torch.randn(10, 6)
    with pytest.raises(ValueError) as exc_info:
        td.LULayer(x.size())
    assert exc_info.type == ValueError
