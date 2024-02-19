import torchdecomp as td
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import sys


def test_create_dummy_matrix():
    size = td.create_dummy_matrix(
        torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])).size()
    assert size[0] == 8
    assert size[1] == 3


def test_print_named_parameters():
    class MLPNet (nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1 * 28 * 28, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, 10)
            self.dropout1 = nn.Dropout2d(0.2)
            self.dropout2 = nn.Dropout2d(0.2)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            return F.relu(self.fc3(x))
    model = MLPNet()
    captured_output = io.StringIO()
    sys.stdout = captured_output
    td.print_named_parameters(model.named_parameters())
    sys.stdout = sys.__stdout__
    printed_message = captured_output.getvalue().strip()
    assert printed_message != ""
