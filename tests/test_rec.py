import torchdecomp as td
import torch
import pytest
import numpy as np


def test_RecLayer():
    x = torch.randn(10, 6)
    rec_layer = td.RecLayer(x, 3)
    assert rec_layer.V.size()[0] == 6
    assert rec_layer.V.size()[1] == 3


def test_RecLayer_error():
    x = np.random.rand(10, 6)
    with pytest.raises(AssertionError) as exc_info:
        td.RecLayer(x, 3)
    assert exc_info.type == AssertionError
