# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics


def accuracy(output, target):
    return metrics.accuracy_score(y_true=target, y_pred=output)


def balanced_accuracy(output, target):
    if np.sum(target) == 0:
        return np.sum(output == 0) / len(output)
    return metrics.balanced_accuracy_score(y_true=target, y_pred=output)


def precision(output, target):
    if np.sum(output) == 0:
        return 0
    return metrics.precision_score(y_true=target, y_pred=output)


def recall(output, target):
    if np.sum(target) == 0:
        return 0
    return metrics.recall_score(y_true=target, y_pred=output)


def f1(output, target):
    if np.sum(target) == 0:
        return 0
    return metrics.f1_score(y_true=target, y_pred=output)


def mcc(output, target):
    if np.sum(target) == 0 or np.sum(output) == 0:
        return 0
    return metrics.matthews_corrcoef(y_true=target, y_pred=output)
