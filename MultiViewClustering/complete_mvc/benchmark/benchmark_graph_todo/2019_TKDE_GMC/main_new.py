import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import csgraph
from scipy.io import loadmat, savemat
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix, normalized_mutual_info_score, adjusted_rand_score

########################################################################################
### Evaluation metrics
########################################################################################
def clustering_acc(y_true, y_pred):
    """Clustering accuracy via Hungarian algorithm."""
    y_true, y_pred = y_true.astype(np.int64), y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.array([[np.sum((y_pred == i) & (y_true == j)) for j in range(D)] for i in range(D)], dtype=np.int64)
    row, col = linear_sum_assignment(w.max() - w)
    return sum(w[r, c] for r, c in zip(row, col)) * 1.0 / y_pred.shape[0]

def purity_score(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = clustering_acc(label, pred)
    pur = purity_score(label, pred)
    return nmi, ari, acc, pur

########################################################################################
### Core math utilities
########################################################################################
def eig1(A, c=None, isMax=1, isSym=1):
    """Compute top-c eigenvectors of A. isMax=0 for smallest eigenvalues."""
    n = A.shape[0]
    c = min(c or n, n)
    if isSym == 1:
        A = np.maximum(A, A.T)
    d, v = np.linalg.eigh(A)
    idx = np.argsort(d) if isMax == 0 else np.argsort(d)[::-1]
    return v[:, idx[:c]], d[idx[:c]], d[idx]

def project_simplex(q0, m=1):
    """Solve: min 1/2 sum_v ||s - q_v||^2  s.t. s >= 0, 1's = 1"""
    ft = 1
    n = q0.shape[1]
    p0 = np.sum(q0, axis=0) / m - np.mean(np.sum(q0, axis=0)) / m + 1 / n
    if np.min(p0) < 0:
        f, lambda_m = 1.0, 0.0
        while np.abs(f) > 1e-10:
            v1 = lambda_m - p0
            posidx = v1 > 0
            g = np.sum(posidx) / n - 1
            if g == 0:
                g = np.finfo(float).eps
            f = np.sum(v1[posidx]) / n - lambda_m
            lambda_m -= f / g
            ft += 1
            if ft > 100:
                break
        x = np.maximum(-v1, 0)
    else:
        x = p0.copy()
    return x, ft

def squared_euclidean_distance(a, b):
    """
    Squared Euclidean distance: ||a_i - b_j||^2.
    a, b: (d, n) matrices, each column is a data point. Returns (n, n).
    """
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    if a.shape[0] == 1:
        a = np.vstack([a, np.zeros((1, a.shape[1]))])
        b = np.vstack([b, np.zeros((1, b.shape[1]))])
    aa = np.sum(a * a, axis=0)
    bb = np.sum(b * b, axis=0)
    d = aa[:, None] + bb[None, :] - 2 * (a.T @ b)
    d = np.maximum(np.real(d), 0)
    d -= np.diag(np.diag(d))
    return d

def init_knn_graph(X, k=5, symmetric=True):
    """
    Initialize kNN graph (Similarity-Induced Graph).
    X: (d, n), each column is a data point. k: number of neighbors.
    Ref: Nie et al., The Constrained Laplacian Rank Algorithm, AAAI 2016.
    """
    _, n = X.shape
    D = squared_euclidean_distance(X, X)
    idx = np.argsort(D, axis=1)
    S = np.zeros((n, n))
    for i in range(n):
        id_ = idx[i, 1:k+2]
        di = D[i, id_]
        S[i, id_] = (di[k] - di) / (k * di[k] - np.sum(di[:k]) + np.finfo(float).eps)
    if symmetric:
        S = (S + S.T) / 2
    return S, D

########################################################################################
### GMC algorithm
########################################################################################
# Graph-based Multi-view Clustering (GMC)
# min sum_v { sum_i ||x_i-x_j||^2 s_ij + alpha||s_i||^2 + w_v||U-S^v||^2 } + lambda*tr(F'L_U F)
# s.t. S^v>=0, 1'S^v_i=1, U>=0, 1'U_i=1, F'F=I
def GMC(X, c, lambda_param=1, normData=1):
    """
    X: list of (d_v, n) arrays, each column is a data point.
    c: number of clusters.
    Returns: y (labels), U (unified graph), S (learned view graphs), S_init, F (embedding), evs
    """
    NITER, zr, pn = 20, 10e-11, 15
    X = [np.array(x.todense()) if sparse.issparse(x) else x.copy() for x in X]
    num = X[0].shape[1]  # n_samples
    m = len(X)            # n_views
    # Z-score normalization
    if normData == 1:
        for v in range(m):
            for j in range(num):
                std = np.std(X[v][:, j])
                if std == 0:
                    std = np.finfo(float).eps
                X[v][:, j] = (X[v][:, j] - np.mean(X[v][:, j])) / std

    # 1. Initialize view-specific graphs S^v
    S = []
    for v in range(m):
        S_v, _ = init_knn_graph(X[v], pn, symmetric=False)
        S.append(S_v)
    S_init = [s.copy() for s in S]

    # 2. Initialize unified graph U (average + row-normalize)
    U = sum(S) / m
    for j in range(num):
        rs = np.sum(U[j, :])
        if rs > 0:
            U[j, :] /= rs

    # 3. Initialize F from Laplacian of U
    sU = (U + U.T) / 2
    L = np.diag(np.sum(sU, axis=1)) - sU
    F, _, eigval_full = eig1(L, c, 0)
    evs = np.zeros((num, NITER + 1))
    evs[:, 0] = eigval_full
    w = np.ones(m) / m

    # Precompute per-view distances and neighbor indices
    ed = [squared_euclidean_distance(X[v], X[v]) for v in range(m)]
    idxx = [np.argsort(ed[v], axis=1) for v in range(m)]

    # 4. Alternating optimization
    for it in range(NITER):
        # (a) Update S^v
        for v in range(m):
            S[v] = np.zeros((num, num))
            for i in range(num):
                id_ = idxx[v][i, 1:pn+2]
                di = ed[v][i, id_]
                numer = di[-1] - di + 2*w[v]*U[i, id_] - 2*w[v]*U[i, id_[-1]]
                denom = (pn*di[-1] - np.sum(di[:-1])+ 2*w[v]*np.sum(U[i, id_[:-1]]) - 2*pn*w[v]*U[i, id_[-1]] + np.finfo(float).eps)
                S[v][i, id_] = np.maximum(numer / denom, 0)

        # (b) Update w
        for v in range(m):
            distUS = np.linalg.norm(U - S[v], 'fro') ** 2
            if distUS == 0:
                distUS = np.finfo(float).eps
            w[v] = 0.5 / np.sqrt(distUS)

        # (c) Update U
        dist = squared_euclidean_distance(F.T, F.T)
        U = np.zeros((num, num))
        for i in range(num):
            idx_list = [i]
            for v in range(m):
                idx_list.extend(np.where(S[v][i, :] > 0)[0].tolist())
            idxs = np.unique(idx_list[1:]) if len(idx_list) >= 1 else np.array([], dtype=int)
            if len(idxs) == 0:
                continue
            q = np.zeros((m, len(idxs)))
            for v in range(m):
                q[v, :] = S[v][i, idxs] - 0.5 * lambda_param / (m * w[v]) * dist[i, idxs]
            x, _ = project_simplex(q, m)
            U[i, idxs] = x

        # (d) Update F
        sU = (U + U.T) / 2
        L = np.diag(np.sum(sU, axis=1)) - sU
        F_old = F.copy()
        F, _, ev = eig1(L, c, 0, 0)
        evs[:len(ev), it + 1] = ev

        # (e) Update lambda and check convergence
        fn1, fn2 = np.sum(ev[:c]), np.sum(ev[:c+1])
        if fn1 > zr:
            lambda_param *= 2
        elif fn2 < zr:
            lambda_param /= 2
            F = F_old
        else:
            print(f'iter = {it + 1}  lambda: {lambda_param}')
            break

    # 5. Final clustering via connected components
    sU = (U + U.T) / 2
    n_components, y = csgraph.connected_components(sparse.csr_matrix(sU), directed=False)
    if n_components != c:
        print(f'Can not find the correct cluster number: {c}')
    return y, U, S, S_init, F, evs

########################################################################################
### Visualization
########################################################################################
def plot_graph(X_views, labels, graph_matrices, prefix, marker_size=10):
    """
    Plot data points (colored by label) with optional graph edges for each view.
    X_views:         list of (n, 2) arrays, one per view.
    labels:          (n,) ground-truth label array.
    graph_matrices:  list of (n, n) weight matrices (one per view), or None for no edges.
    prefix:          filename prefix -> saved as '{prefix}_view{v}.png'.
    """
    cLab = np.unique(labels)
    edge_color = [0, 197/255, 205/255]
    gray = [79/255, 79/255, 79/255]
    for v, Xv in enumerate(X_views):
        n = Xv.shape[0]
        plt.figure(figsize=(5, 5))
        plt.plot(Xv[:, 0], Xv[:, 1], '.k', markersize=marker_size)
        plt.plot(Xv[labels == cLab[0], 0], Xv[labels == cLab[0], 1], '.r', markersize=marker_size)
        plt.plot(Xv[labels == cLab[1], 0], Xv[labels == cLab[1], 1], '.', markersize=marker_size)
        if len(cLab) > 2:
            plt.plot(Xv[labels == cLab[2], 0], Xv[labels == cLab[2], 1], '.', color=gray, markersize=marker_size)
        if graph_matrices is not None:
            G = graph_matrices[v]
            for ii in range(n):
                for jj in range(ii + 1):
                    wt = G[ii, jj]
                    if wt > 0:
                        plt.plot([Xv[ii, 0], Xv[jj, 0]], [Xv[ii, 1], Xv[jj, 1]], '-', color=edge_color, linewidth=5 * wt)
        plt.axis('equal')
        plt.savefig(f'./result/{prefix}_view{v}.png')
        plt.close()

########################################################################################
### Main: Toy examples (TwoMoon / ThreeRing)
########################################################################################
def load_toy_data(dataname, datadir='data/'):
    """Load toy multi-view data from .mat file. Returns X (list of (n,d) arrays), y0 (1D labels)."""
    mat = loadmat(os.path.join(datadir, dataname))
    if 'X1' in mat:
        X = [mat['X1'], mat['X2']]
    else:
        X = [mat['X'][0, 0], mat['X'][0, 1]]
    for i in range(len(X)):
        if isinstance(X[i], np.ndarray) and X[i].dtype == np.object_:
            X[i] = X[i].flat[0]
    y0 = mat.get('y0', mat.get('Y'))
    if y0.ndim > 1 and y0.shape[0] < y0.shape[1]:
        y0 = y0.T
    y0 = y0[:, 0] if y0.ndim > 1 else y0.ravel()
    return X, y0

def main_toy(dataname='TwoMoon'):
    assert dataname in ('TwoMoon', 'ThreeRing')
    c = 2 if dataname == 'TwoMoon' else 3
    # 1. Load data
    X, y0 = load_toy_data(dataname)
    data = [Xv.T for Xv in X]  # GMC expects (d, n)
    # 2. Run GMC (normData=0 for toy data)
    predY, U, S, S_init, F, evs = GMC(data, c, lambda_param=1, normData=0)
    # 3. Evaluate
    nmi, ari, acc, pur = evaluate(y0, predY)
    print(f'{dataname} -> ACC:{acc:.4f}  NMI:{nmi:.4f}  ARI:{ari:.4f}  Purity:{pur:.4f}')
    # 4. Visualize: original data, initial graph, learned graph, unified graph
    plot_graph(X, y0, None,            'toy_original')
    plot_graph(X, y0, S_init,          'toy_initial_graph')
    plot_graph(X, y0, S,               'toy_learned_graph')
    plot_graph(X, y0, [U] * len(X),   'toy_unified_graph')
    print('Figures saved.')

########################################################################################
### Main: Real-world datasets
########################################################################################
def main_real_world(datanames=None, runtimes=1):
    datanames = datanames or ['100leaves']
    datadir, resultdir = 'data/', 'result/'
    os.makedirs(resultdir, exist_ok=True)
    for name in datanames:
        if not os.path.isfile(os.path.join(datadir, name + '.mat')):
            print(f'Skip {name}: file not found')
            continue
        mat = loadmat(os.path.join(datadir, name))
        X = mat['data']
        X = [X.flat[i] for i in range(X.size)] if X.dtype == np.object_ else [X]
        y0 = mat['truelabel']
        y0 = y0.flat[0].ravel() if y0.dtype == np.object_ else y0.ravel()
        c = len(np.unique(y0))
        ACC, NMI, ARI = np.zeros(runtimes), np.zeros(runtimes), np.zeros(runtimes)
        for t in range(runtimes):
            y, U, S, S_init, F, evs = GMC(X, c)
            nmi, ari, acc, pur = evaluate(y0, y)
            ACC[t], NMI[t], ARI[t] = acc, nmi, ari
            print(f'{name}  run {t+1}/{runtimes}')
            print(f'ACC:{acc:.7f}\tNMI:{nmi:.7f}\tARI:{ari:.7f}\tPurity:{pur:.7f}')
        ncol = max(3, runtimes)
        Result = np.zeros((5, ncol))
        Result[0, :runtimes] = ACC
        Result[1, :runtimes] = NMI
        Result[2, :runtimes] = ARI
        Result[3, :3] = [np.mean(ACC), np.mean(NMI), np.mean(ARI)]
        Result[4, :3] = [np.std(ACC), np.std(NMI), np.std(ARI)]
        outpath = os.path.join(resultdir, f'{name}_result.mat')
        savemat(outpath, {'Result': Result, 'U': U, 'y0': y0, 'y': y})
        print(f'Saved {outpath}')

if __name__ == "__main__":
    # main_toy('TwoMoon')
    # main_toy('ThreeRing')
    main_real_world(['100leaves'])
    # main_real_world(['3sources'])
    # main_real_world(['BBC'])
    # main_real_world(['BBCSport'])
    # main_real_world(['HW'])
    # main_real_world(['HW2sources'])
    # main_real_world(['NGs'])
    # main_real_world(['WebKB'])
    # main_real_world(['Hdigit'])
    # main_real_world(['Mfeat'])
