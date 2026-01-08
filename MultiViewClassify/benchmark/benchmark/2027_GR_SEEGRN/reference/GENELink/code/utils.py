# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn import metrics
from thop import profile, clever_format

def calculate_auroc(predictions, labels):
    """
    Calculate auroc.
    :param predictions: predictions
    :param labels: labels
    :return: fpr_list, tpr_list, auroc
    """
    if np.max(labels) ==1 and np.min(labels)==0:
        fpr_list, tpr_list, _ = metrics.roc_curve(y_true=labels, y_score=predictions, drop_intermediate=True)
        auroc = metrics.roc_auc_score(labels, predictions)
    else:
        fpr_list, tpr_list = [], []
        auroc = np.nan
    return fpr_list, tpr_list, auroc


def calculate_aupr(predictions, labels):
    """
    Calculate aupr.
    :param predictions: predictions
    :param labels: labels
    :return: precision_list, recall_list, aupr
    """
    if np.max(labels) == 1 and np.min(labels) == 0:
        precision_list, recall_list, _ = metrics.precision_recall_curve(y_true=labels, probas_pred=predictions)
        aupr = metrics.average_precision_score(labels, predictions)
    else:
        precision_list, recall_list = [], []
        aupr = np.nan
    return precision_list, recall_list, aupr


def calculate_ep(predictions, labels):
    """
    Calculate early precision and early precision ratio.
    :param predictions: predictions
    :param labels: labels
    :return: ep
    """
    if np.max(labels) == 1 and np.min(labels) == 0:
        k = int(np.sum(labels))
        sorted_index = np.argsort(predictions)[::-1] # 
        sorted_labels = labels[sorted_index]
        sorted_labels_top_k = sorted_labels[:k]
        ep = np.sum(sorted_labels_top_k) / k
        density = k / len(labels)
        epr = ep / density
    else:
        ep = np.nan
        epr = np.nan
    return ep, epr


def model_complexity(model, inputs):
    macs, params = profile(model, inputs=inputs, verbose=False)
    params = sum(p.numel() for p in model.parameters())
    macs, params = clever_format([macs, params], "%.3f")
    return macs, params
