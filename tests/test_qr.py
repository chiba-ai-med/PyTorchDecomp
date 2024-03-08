import torchdecomp as td
import torch
import pytest
import numpy as np


def test_QRLayer():
    x = torch.randn(6, 6)
    qr_layer = td.QRLayer(x)
    assert qr_layer.Q.size()[0] == 6
    assert qr_layer.Q.size()[1] == 6
    assert qr_layer.R.size()[0] == 6
    assert qr_layer.R.size()[1] == 6


def test_QRLayer_error():
    x = torch.randn(10, 6)
    with pytest.raises(AssertionError) as exc_info:
        td.QRLayer(x)
    assert exc_info.type == AssertionError


def test_QRLayer_error2():
    x = np.random.rand(6, 6)
    with pytest.raises(AssertionError) as exc_info:
        td.QRLayer(x)
    assert exc_info.type == AssertionError
