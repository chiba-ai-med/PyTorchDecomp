import torchdecomp as td
import torch


def test_FactorLayer():
    x = torch.randn(10, 6)
    factor_layer = td.FactorLayer(x.size(1), 3)
    assert factor_layer.V.size()[0] == 6
    assert factor_layer.V.size()[1] == 3
