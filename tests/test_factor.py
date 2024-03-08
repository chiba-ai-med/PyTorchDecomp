import torchdecomp as td
import torch
import pytest
import numpy as np


def test_FactorLayer():
    x = torch.randn(10, 6)
    factor_layer = td.FactorLayer(x, 3)
    assert factor_layer.V.size()[0] == 6
    assert factor_layer.V.size()[1] == 3


def test_FactorLayer_error():
    x = np.random.rand(10, 6)
    with pytest.raises(AssertionError) as exc_info:
        td.FactorLayer(x, 3)
    assert exc_info.type == AssertionError
