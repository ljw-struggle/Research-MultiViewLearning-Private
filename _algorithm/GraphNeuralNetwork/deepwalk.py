# -*- coding: utf-8 -*-
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression

from ge import DeepWalk
from ge.classify import read_node_label, Classifier


def evaluate_embeddings(embeddings):
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings,):
    X, Y = read_node_label('./data/wiki_labels.txt')

    embedding_list = []
    for k in X:
        embedding_list.append(embeddings[k])
    embedding_list = np.array(embedding_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(embedding_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.savefig('./result.png')


if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    G = nx.read_edgelist('./data/StringDB/edge.txt', create_using=nx.DiGraph())

    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    model.train(window_size=5, iter=3)
    embeddings = model.get_embeddings()

    mTORC1_list = ['AKT1', 'AKT1S1', 'AKT2', 'AKT3', 'BMT2', 'C12orf66', 'CASTOR1', 'CASTOR2', 'DDIT4', 'DEPDC5', 
               'DEPTOR', 'FLCN', 'FNIP1', 'FNIP2', 'ITFG2', 'KPTN', 'LAMTOR1', 'LAMTOR2', 'LAMTOR3', 'LAMTOR4',
               'LAMTOR5', 'LARS', 'MIOS', 'MLST8', 'MTOR', 'NPRL2', 'RHEB', 'RPTOR', 'RRAGA', 'RRAGB', 
               'RRAGC', 'RRAGD', 'SAR1A', 'SAR1B', 'SEC13', 'SEH1L', 'SESN1', 'SESN2', 'SESN3', 'SLC38A9', 
               'SZT2', 'TBC1D7', 'TSC1', 'TSC2', 'WDR24', 'WDR59']

    label = []
    for name in embeddings.keys():
        if name in mTORC1_list:
            label.append(1)
        else:
            label.append(0)

    np.savez('./data/stringdb.npz', name=list(embeddings.keys()), data=list(embeddings.values()), label=label)


    # evaluate_embeddings(embeddings)
    # plot_embeddings(embeddings)
