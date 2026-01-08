# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


class CrossEntropy(torch.nn.CrossEntropyLoss):
    """
    Cross Entropy Loss.
    Implementation of cross entropy loss. (<https://pytorch.org/docs/stable/nn.html#crossentropyloss>)(PyTorch)
    Cross Entropy Loss Formula: CE = - y_true * log(y_pred)
                        ,which y_pred = softmax(logits), y_true = target_tensor.
    """
    def __init__(self,
                 weight=None,
                 device='cpu'):
        """
        Initializes CrossEntropyLoss class and sets attributes needed in loss calculation.
        :param weight: tensor
        :param size_average: bool
        """
        weight = torch.tensor(weight, dtype=torch.float32).to(device)
        super(CrossEntropy, self).__init__(weight=weight)


class BinaryCrossEntropy(torch.nn.Module):
    """
    Cross Entropy Loss.
    Implementation of cross entropy loss. (<https://pytorch.org/docs/stable/nn.html#crossentropyloss>)(PyTorch)
    Cross Entropy Loss Formula: CE = - y_true * alpha * log(y_pred) - (1 - y_true) * (1-alpha) * log(1 - y_pred)
                        ,which y_pred = sigmoid(logits), y_true = target_tensor.
    """
    def __init__(self,
                 alpha=0.5,
                 size_average=True):
        """
        Initializes CrossEntropyLoss class and sets attributes needed in loss calculation.
        :param weight: tensor
        :param size_average: bool
        """
        super(BinaryCrossEntropy, self).__init__()
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, pred, target):
        """
        Computes binary cross entropy loss between predicted probabilities and true label.
        :param pred: predicted probabilities (sigmoid). shape = (batch_size, ..., 1)
        :param target: ground truth labels. shape = (batch_size, ..., 1)
        :return: cross entropy loss.
        """
        target = target.float()

        # 1\ Clip values for Numerical Stable. (Avoid NaN calculations)
        epsilon = 1e-07
        pred = pred.clamp(min=epsilon, max=1.0 - epsilon)

        # 2\ Calculate the cross entropy loss.
        loss =  target * self.alpha * torch.log(pred)
        loss += (1 - target) * (1 - self.alpha) * torch.log(1 - pred)
        loss = -loss

        # 3\ Size Average.
        if self.size_average:
            loss = torch.mean(loss)
        else:
            loss = torch.sum(loss)
        return loss


class BinaryFocalLoss(torch.nn.Module):
    """
    Focal Loss.
    Implementation of focal loss. (<https://arxiv.org/pdf/1708.02002.pdf>)(Kaiming He)
    Focal Loss Formula: FL = - y_true * alpha * (1-y_pred)^gamma * log(y_pred)
                             - (1 - y_true) * (1-alpha) * y_pred^gamma * log(1-y_pred)
                        ,which alpha = 0.25(default), gamma = 2(default), y_pred = sigmoid(logits), y_true = target_tensor.
    """
    def __init__(self,
                 alpha=0.5,
                 gamma=2.0,
                 size_average=True):
        """
        Initializes FocalLoss class and sets attributes needed in loss calculation.
        :param alpha: float
        :param gamma: float
        :param size_average: bool
        """
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        """
        Computes binary focal loss between predicted probabilities and true label.
        :param pred: predicted probabilities (sigmoid). shape = (batch_size, ..., 1)
        :param target: ground truth labels. shape = (batch_size, ..., 1)
        :return: focal loss.
        """
        target = target.float()

        # 1\ Clip values for Numerical Stable. (Avoid NaN calculations)
        epsilon = 1e-07
        pred = pred.clamp(min=epsilon, max=1.0 - epsilon)

        # 2\ Calculate the focal loss.
        focal = target * self.alpha * torch.pow(1 - pred, self.gamma) * torch.log(pred)
        focal += (1 - target) * (1 - self.alpha) * torch.pow(pred, self.gamma) * torch.log(1 - pred)
        loss = -focal

        # 3\ Size Average.
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
