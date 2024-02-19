import sys
import os
import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


# Helper functions
def is_symmetric_matrix(x):
    if len(x) != len(x[0]):
        check1 = False
    else:
        check1 = True
    
    if all(x[i][j] == x[j][i] for i in range(len(x)) for j in range(len(x))):
        check2 = True
    else:
        check2 = False
    
    return check1 & check2


def create_dummy_matrix(class_vector):
    """Creates a dummy matrix from a class label vector.

    Args:
        class_vector: A PyTorch array with numeric elements

    Returns:
        A PyTorch array filled with dummy vectors

    Example:
        >>> import torchdecomp as td
        >>> td.create_dummy_matrix(torch.tensor([0, 1, 2, 1, 0, 2, 1, 0]))

    Note:
       The number of rows is the number of classes
       and the number of columns is the number of data.

    """
    unique_classes = torch.unique(class_vector)
    num_data = len(class_vector)
    num_classes = len(unique_classes)
    dummy_matrix = torch.zeros((num_data, num_classes), dtype=torch.float32)
    for i, class_label in enumerate(unique_classes):
        class_indices = (class_vector == class_label).nonzero().view(-1)
        dummy_matrix[class_indices, i] = 1.0
    return dummy_matrix


def print_named_parameters(named_params):
    """Outputs the contents of the named parameters.

    Args:
        named_params: An object instantiated by user's original class
        defined from PyTorch's nn.Module.

    Returns:
        Leaf variables defined as PyTorch Tensor(s)
        set with requires_grad_(), requires_grad=True option,
        or nn.Parameter (cf. nn.Module).

    Example:
        >>> import torchdecomp as td
        >>> import torch
        >>> import torch.nn as nn
        >>> import torch.nn.functional as F
        >>> class MLPNet (nn.Module):
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
        >>> model = MLPNet()
        >>> td.print_named_parameters(model.named_parameters())

    Note:
       These Tensor object(s) is/are subject to the optimization
       by gradient descent (e.g., torch.optim.SGD)

    """
    for name, param in named_params:
        print(f"{name}: {param.size()}")


# Disable
def _blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def _enablePrint():
    sys.stdout = sys.__stdout__


def _rho(beta, root=False):
    if root:
        out = 0.5
    else:
        if beta < 1:
            out = 1 / (2 - beta)
        if (1 <= beta) & (beta <= 2):
            out = 1
        if beta > 2:
            out = 1 / (beta - 1)
    return out


def _plot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(
                    img, masks.to(torch.bool),
                    colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
