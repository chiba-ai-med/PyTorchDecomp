import torch
import torch.nn as nn


class NMFLayer(nn.Module):
    """Non-negative Matrix Factorization Layer

    A non-negative matrix X (n times m) is decomposed to
    the product of W (n times k) and H (k times m).

    Attributes:
        size (int): The size of X (n times m)
        n_components (int): The number of lower dimensions (k)
        l1_lambda_w (float): L1 regularization parameter for W
        l1_lambda_h (float): L1 regularization parameter for H
        l2_lambda_w (float): L2 regularization parameter for W
        l2_lambda_h (float): L2 regularization parameter for H
        bin_lambda_w (float): Binarization regularization parameter for W
        bin_lambda_h (float): Binarization regularization parameter for H
        eps (float): Offset value to avoid zero division
        beta (float): Beta parameter of Beta-divergence

    Example:
        >>> import torchdecomp as td
        >>> import torch
        >>> torch.manual_seed(123456)
        >>> x = torch.randn(10, 6) # Test datasets
        >>> nmf_layer = td.NMFLayer(x.size(), 3) # Instantiation

    """
    def __init__(
        self, size, n_components,
        l1_lambda_w=torch.finfo(torch.float64).eps,
        l1_lambda_h=torch.finfo(torch.float64).eps,
        l2_lambda_w=torch.finfo(torch.float64).eps,
        l2_lambda_h=torch.finfo(torch.float64).eps,
        bin_lambda_w=torch.finfo(torch.float64).eps,
        bin_lambda_h=torch.finfo(torch.float64).eps,
            eps=torch.finfo(torch.float64).eps, beta=2):
        """Initialization function
        """
        super(NMFLayer, self).__init__()
        self.eps = eps
        self.W = nn.Parameter(torch.rand(
            size[0], n_components, dtype=torch.float64))
        self.H = nn.Parameter(torch.rand(
            n_components, size[1], dtype=torch.float64))
        self.l1_lambda_w = l1_lambda_w
        self.l1_lambda_h = l1_lambda_h
        self.l2_lambda_w = l2_lambda_w
        self.l2_lambda_h = l2_lambda_h
        self.bin_lambda_w = bin_lambda_w
        self.bin_lambda_h = bin_lambda_h
        self.beta = beta
    
    def positive(self, X, WH, beta):
        """Positive Terms of Beta-NMF Object Function
        """
        if beta == 0:
            return X / WH
        if beta == 1:
            return (1 / (beta + 0.001)) * (WH**(beta + 0.001))
        else:
            return (1 / beta) * (WH**beta)
    
    def negative(self, X, WH, beta):
        """Negative Terms of Beta-NMF Object Function
        """
        if beta == 0:
            return torch.log(X / WH)
        if beta == 1:
            return (1 / (beta - 0.999)) * (X * (WH**(beta - 0.999)))
        else:
            return (1 / (beta - 1)) * (X * (WH**(beta - 1)))
    
    def positive_w(self, W, l1_lambda_w, l2_lambda_w, bin_lambda_w):
        """Positive Terms of L2 regularization against W
        """
        l1_term = l1_lambda_w * W
        l2_term = l2_lambda_w * W**2
        bin_term = bin_lambda_w * (W**4 + W**2)
        return l1_term + l2_term + bin_term
    
    def negative_w(self, W, bin_lambda_w):
        """Negative Terms of L2 regularization against W
        """
        bin_term = bin_lambda_w * 2 * W**3
        return bin_term
    
    def positive_h(self, H, l1_lambda_h, l2_lambda_h, bin_lambda_h):
        """Positive Terms of L2 regularization against H
        """
        l1_term = l1_lambda_h * H
        l2_term = l2_lambda_h * H**2
        bin_term = bin_lambda_h * (H**4 + H**2)
        return l1_term + l2_term + bin_term
    
    def negative_h(self, H, bin_lambda_h):
        """Negative Terms of L2 regularization against H
        """
        bin_term = bin_lambda_h * 2 * H**3
        return bin_term
    
    def loss(self, pos, neg, pos_w, neg_w, pos_h, neg_h):
        """Total Loss with the recontruction term and regularization terms
        """
        loss1 = torch.sum(pos - neg)
        loss2 = torch.sum(pos_w - neg_w)
        loss3 = torch.sum(pos_h - neg_h)
        return loss1 + loss2 + loss3

    def forward(self, X):
        """Forward propagation function
        """
        WH = torch.mm(self.W, self.H)
        WH[WH < self.eps] = self.eps
        pos = self.positive(X, WH, self.beta)
        neg = self.negative(X, WH, self.beta)
        pos_w = self.positive_w(
            self.W, self.l1_lambda_w,
            self.l2_lambda_w, self.bin_lambda_w)
        neg_w = self.negative_w(self.W, self.bin_lambda_w)
        pos_h = self.positive_h(
            self.H, self.l1_lambda_h,
            self.l2_lambda_h, self.bin_lambda_h)
        neg_h = self.negative_h(self.H, self.bin_lambda_h)
        loss = self.loss(pos, neg, pos_w, neg_w, pos_h, neg_h)
        return loss, WH, pos, neg, pos_w, neg_w, pos_h, neg_h
