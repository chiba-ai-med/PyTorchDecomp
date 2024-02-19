import torchdecomp as td
import torch


def test_SymRecLayer():
    x = torch.randn(6, 6)
    x = torch.mm(x, x.t())
    symrec_layer = td.SymRecLayer(x.size(0), 3)
    assert symrec_layer.U.size()[0] == 6
    assert symrec_layer.U.size()[1] == 3
