import torchdecomp as td
import torch


def test_QRLayer():
    x = torch.randn(6, 6)
    qr_layer = td.QRLayer(x.size())
    assert qr_layer.Q.size()[0] == 6
    assert qr_layer.Q.size()[1] == 6
    assert qr_layer.R.size()[0] == 6
    assert qr_layer.R.size()[1] == 6
