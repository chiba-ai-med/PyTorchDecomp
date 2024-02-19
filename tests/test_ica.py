import torchdecomp as td
import torch


def test_RotationLayer():
    x = torch.randn(6, 6)
    rotation_layer = td.RotationLayer(x.size(0))
    assert rotation_layer.mixing_matrix.size()[0] == 6
    assert rotation_layer.mixing_matrix.size()[1] == 6


def test_KurtosisICALayer():
    x = torch.randn(6, 6)
    rotation_layer = td.RotationLayer(x.size(0))
    x_rotated = rotation_layer(x)
    loss = td.KurtosisICALayer()
    output = loss(x_rotated).data
    assert output is not None
    assert not isinstance(output, (list, tuple, dict, set))


def test_NegentropyICALayer():
    x = torch.randn(6, 6)
    rotation_layer = td.RotationLayer(x.size(0))
    x_rotated = rotation_layer(x)
    loss = td.NegentropyICALayer()
    output = loss(x_rotated).data
    assert output is not None
    assert not isinstance(output, (list, tuple, dict, set))


def test_DDICALayer():
    x = torch.randn(6, 6)
    loss = td.DDICALayer(sigma=0.01, alpha=0.75, size=x.size(0))
    output = loss(x)
    assert output is not None
    assert not isinstance(output, (list, tuple, dict, set))
