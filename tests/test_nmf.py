import torchdecomp as td
import torch


def test_NMFLayer():
    x = torch.randn(10, 6)
    nmf_layer = td.NMFLayer(x.size(), 3)
    assert nmf_layer.W.size()[0] == 10
    assert nmf_layer.W.size()[1] == 3
    assert nmf_layer.H.size()[0] == 3
    assert nmf_layer.H.size()[1] == 6
