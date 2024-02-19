import torchdecomp as td
import torch


def test_RecLayer():
    x = torch.randn(10, 6)
    rec_layer = td.RecLayer(x.size(1), 3)
    assert rec_layer.V.size()[0] == 6
    assert rec_layer.V.size()[1] == 3
