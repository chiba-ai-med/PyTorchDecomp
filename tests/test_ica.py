import torchdecomp as td
import torch
import pytest
import numpy as np


def test_RotationLayer():
    x = torch.randn(10, 6)
    rotation_layer = td.RotationLayer(x)
    assert rotation_layer.mixing_matrix.size()[0] == 6
    assert rotation_layer.mixing_matrix.size()[1] == 6


def test_KurtosisICALayer():
    x = torch.randn(10, 6)
    rotation_layer = td.RotationLayer(x)
    x_rotated = rotation_layer(x)
    loss = td.KurtosisICALayer()
    output = loss(x_rotated).data
    assert output is not None
    assert not isinstance(output, (list, tuple, dict, set))


def test_NegentropyICALayer():
    x = torch.randn(10, 6)
    rotation_layer = td.RotationLayer(x)
    x_rotated = rotation_layer(x)
    loss = td.NegentropyICALayer()
    output = loss(x_rotated).data
    assert output is not None
    assert not isinstance(output, (list, tuple, dict, set))


def test_DDICALayer():
    x = torch.randn(10, 6)
    loss = td.DDICALayer(x=x, sigma=0.01, alpha=0.75)
    output = loss(x)
    assert output is not None
    assert not isinstance(output, (list, tuple, dict, set))
