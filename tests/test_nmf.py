import torchdecomp as td
import torch
import pytest
import numpy as np


def test_NMFLayer():
    x = torch.randn(10, 6)
    nmf_layer = td.NMFLayer(x, 3)
    assert nmf_layer.W.size()[0] == 10
    assert nmf_layer.W.size()[1] == 3
    assert nmf_layer.H.size()[0] == 3
    assert nmf_layer.H.size()[1] == 6


def test_NMFLayer_error():
    x = np.random.rand(10, 6)
    with pytest.raises(AssertionError) as exc_info:
        td.NMFLayer(x, 3)
    assert exc_info.type == AssertionError
