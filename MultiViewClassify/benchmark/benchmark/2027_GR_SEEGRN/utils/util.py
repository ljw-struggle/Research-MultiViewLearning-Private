# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from thop import profile, clever_format
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from sklearn import metrics
from datetime import datetime


class Timer(object):
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()

    def now(self):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S %p')
        return now


class MetricTracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def result(self, key=None):
        result = dict(self._data.average)
        return result if key is None else result[key]


def ensure_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def create_dirs(dirs):
    """ Create dirs. (recurrent) """
    for path in dirs:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=False)


def model_complexity(model, inputs):
    macs, params = profile(model, inputs=inputs, verbose=False)
    params = sum(p.numel() for p in model.parameters())
    macs, params = clever_format([macs, params], "%.3f")
    return macs, params


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


def plot_roc_curve(fpr_list, tpr_list, file_path):
    plt.figure()
    plt.plot(fpr_list, tpr_list, lw=1, linestyle='-')
    plt.plot([0, 1], [0, 1], color='navy', lw=0.2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(u"Receiver Operating Characteristic Curve")
    plt.savefig(file_path)


def plot_pr_curve(precision_list, recall_list, file_path):
    plt.figure()
    plt.plot(recall_list, precision_list, lw=1, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(u"Precision Recall Curve")
    plt.savefig(file_path)
