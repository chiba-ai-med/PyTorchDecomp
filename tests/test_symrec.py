import torchdecomp as td
import torch
import pytest
import numpy as np


def test_SymRecLayer():
    x = torch.randn(6, 6)
    x = torch.mm(x, x.t())
    symrec_layer = td.SymRecLayer(x, 3)
    assert symrec_layer.U.size()[0] == 6
    assert symrec_layer.U.size()[1] == 3


def test_SymRecLayer_error():
    x = torch.randn(6, 6)
    with pytest.raises(AssertionError) as exc_info:
        td.SymRecLayer(x, 3)
    assert exc_info.type == AssertionError


def test_SymRecLayer_error2():
    x = np.random.rand(6, 6)
    x = np.matmul(x, x.T)
    with pytest.raises(AssertionError) as exc_info:
        td.SymRecLayer(x, 3)
    assert exc_info.type == AssertionError
