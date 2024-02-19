import torchdecomp as td
import torch


def test_CholeskyLayer():
    x = torch.randn(6, 6)
    x = torch.mm(x, x.t())
    cholesky_layer = td.CholeskyLayer(x.size())
    assert cholesky_layer.L.size()[0] == 6
    assert cholesky_layer.L.size()[1] == 6
