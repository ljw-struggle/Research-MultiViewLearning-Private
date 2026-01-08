import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import csgraph
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix, normalized_mutual_info_score, adjusted_rand_score, silhouette_score

def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = clustering_acc(label, pred)
    pur = purity_score(label, pred)
    # asw = silhouette_score(embedding, pred) # silhouette score
    return nmi, ari, acc, pur #, asw

def clustering_acc(y_true, y_pred): # y_pred and y_true are numpy arrays, same shape
    y_true = y_true.astype(np.int64); y_pred = y_pred.astype(np.int64); assert y_pred.size == y_true.size; 
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.array([[sum((y_pred == i) & (y_true == j)) for j in range(D)] for i in range(D)], dtype=np.int64) # shape: (num_pred_clusters, num_true_clusters)
    ind = linear_sum_assignment(w.max() - w) # align clusters using the Hungarian algorithm, ind[0] is the row indices (predicted clusters), ind[1] is the column indices (true clusters)
    return sum([w[i][j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.shape[0] # accuracy

def purity_score(y_true, y_pred):
    contingency_matrix_result = contingency_matrix(y_true, y_pred) # shape: (num_true_clusters, num_pred_clusters)
    return np.sum(np.amax(contingency_matrix_result, axis=0)) / np.sum(contingency_matrix_result) 

def eig1(A, c=None, isMax=1, isSym=1):
    # Compute top c eigenvectors of matrix A.
    n = A.shape[0]
    if c is None:
        c = n
    elif c > n:
        c = n

    if isSym == 1:
        A = np.maximum(A, A.T)

    d, v = np.linalg.eigh(A)
    if isMax == 0:
        idx = np.argsort(d)
    else:
        idx = np.argsort(d)[::-1]

    idx1 = idx[:c]
    eigval = d[idx1]
    eigvec = v[:, idx1]
    eigval_full = d[idx]
    return eigvec, eigval, eigval_full

def SloutionToP19(q0, m=1):
    """
    min  1/2 sum_v || s - qv||^2
    s.t. s>=0, 1's=1
    """
    ft = 1
    n = q0.shape[1]
    p0 = np.sum(q0, axis=0) / m - np.mean(np.sum(q0, axis=0)) / m + 1 / n
    vmin = np.min(p0)
    if vmin < 0:
        f = 1
        lambda_m = 0.0
        while np.abs(f) > 1e-10:
            v1 = lambda_m - p0
            posidx = v1 > 0
            npos = np.sum(posidx)
            g = npos / n - 1
            if g == 0:
                g = np.finfo(float).eps
            f = np.sum(v1[posidx]) / n - lambda_m
            lambda_m = lambda_m - f / g
            ft += 1
            if ft > 100:
                x = np.maximum(-v1, 0)
                break
        else:
            x = np.maximum(-v1, 0)
    else:
        x = p0.copy()
    return x, ft


def L2_distance_1(a, b):
    """
    Compute squared Euclidean distance.
    ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
    a, b: two matrices, each column is a data point
    d: distance matrix of a and b
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    if a.shape[0] == 1:
        a = np.vstack([a, np.zeros((1, a.shape[1]))])
        b = np.vstack([b, np.zeros((1, b.shape[1]))])
    aa = np.sum(a * a, axis=0)
    bb = np.sum(b * b, axis=0)
    ab = a.T @ b
    d = np.tile(aa.reshape(-1, 1), (1, bb.size)) + np.tile(bb, (aa.size, 1)) - 2 * ab
    d = np.real(d)
    d = np.maximum(d, 0)
    # force 0 on the diagonal
    d = d - np.diag(np.diag(d))
    return d

def InitializeSIGs(X, k=5, issymmetric=1):
    """
    Initialize SIG (Similarity-Induced Graph) matrices.
    X: each column is a data point
    k: number of neighbors
    issymmetric: set S = (S+S')/2 if issymmetric=1
    Ref: F. Nie, X. Wang, M. I. Jordan, and H. Huang, The constrained
    Laplacian rank algorithm for graph-based clustering, in AAAI, 2016.
    """
    _, n = X.shape
    D = L2_distance_1(X, X)
    idx = np.argsort(D, axis=1)  # sort each row
    S = np.zeros((n, n))
    for i in range(n):
        id_ = idx[i, 1 : k + 2]  # skip self (first column)
        di = D[i, id_]
        denom = k * di[k] - np.sum(di[:k]) + np.finfo(float).eps
        S[i, id_] = (di[k] - di) / denom
    if issymmetric == 1:
        S = (S + S.T) / 2
    return S, D

def GMC(X, c, lambda_param=1, normData=1):
    NITER = 20
    zr = 10e-11
    pn = 15  # number of neighbours for constructS_PNG
    islocal = 1
    num = X[0].shape[1]  # number of instances
    m = len(X)  # number of views
    # Normalization: Z-score
    if normData == 1:
        for i in range(m):
            for j in range(num):
                norm_item = np.std(X[i][:, j])
                if norm_item == 0:
                    norm_item = np.finfo(float).eps
                X[i][:, j] = (X[i][:, j] - np.mean(X[i][:, j])) / norm_item
    # initialize S0: Constructing the SIG matrices
    S0 = []
    for i in range(m):
        S_i, _ = InitializeSIGs(X[i], pn, 0)
        S0.append(S_i)
    S0_initial = [s.copy() for s in S0]
    # initialize U, F and w
    U = np.zeros((num, num))
    for i in range(m):
        U = U + S0[i]
    U = U / m
    for j in range(num):
        row_sum = np.sum(U[j, :])
        if row_sum > 0:
            U[j, :] = U[j, :] / row_sum
    sU = (U + U.T) / 2
    D = np.diag(np.sum(sU, axis=1))
    L = D - sU
    F, _, eigval_full = eig1(L, c, 0)
    evs = np.zeros((num, NITER + 1))
    evs[:, 0] = eigval_full
    w = np.ones(m) / m
    idxx = []
    ed = []
    for v in range(m):
        ed_v = L2_distance_1(X[v], X[v])
        ed.append(ed_v)
        idxx.append(np.argsort(ed_v, axis=1))
    # update ...
    for iter in range(NITER):
        # update S^v
        for v in range(m):
            S0[v] = np.zeros((num, num))
            for i in range(num):
                # id: 2 to pn+2 in MATLAB (1-based) -> 1:pn+2 in Python (0-based indices)
                id_ = idxx[v][i, 1 : pn + 2]
                di = ed[v][i, id_]
                numerator = di[-1] - di + 2 * w[v] * U[i, id_] - 2 * w[v] * U[i, id_[-1]]
                denominator1 = pn * di[-1] - np.sum(di[:-1])
                denominator2 = 2 * w[v] * np.sum(U[i, id_[:-1]]) - 2 * pn * w[v] * U[i, id_[-1]]
                denom = denominator1 + denominator2 + np.finfo(float).eps
                S0[v][i, id_] = np.maximum(numerator / denom, 0)
        # update w
        for v in range(m):
            US = U - S0[v]
            distUS = np.linalg.norm(US, 'fro') ** 2
            if distUS == 0:
                distUS = np.finfo(float).eps
            w[v] = 0.5 / np.sqrt(distUS)
        # update U
        dist = L2_distance_1(F.T, F.T)
        U = np.zeros((num, num))
        for i in range(num):
            # Match MATLAB: idx = [idx, find(s0>0)] per view, then idxs = unique(idx(2:end))
            idx_list = []
            for v in range(m):
                s0 = S0[v][i, :]
                idx_list.extend(np.where(s0 > 0)[0].tolist())
            if len(idx_list) >= 1:
                idxs = np.unique(idx_list[1:])
            else:
                idxs = np.array([], dtype=int)
            if islocal == 1:
                idxs0 = idxs
            else:
                idxs0 = np.arange(num)
            if len(idxs0) == 0:
                continue
            q = np.zeros((m, len(idxs0)))
            for v in range(m):
                s1 = S0[v][i, :]
                si = s1[idxs0]
                di = dist[i, idxs0]
                mw = m * w[v]
                lmw = lambda_param / mw
                q[v, :] = si - 0.5 * lmw * di
            x, _ = SloutionToP19(q, m)
            U[i, idxs0] = x
        # update F
        sU = U.copy()
        sU = (sU + sU.T) / 2
        D = np.diag(np.sum(sU, axis=1))
        L = D - sU
        F_old = F.copy()
        F, _, ev = eig1(L, c, 0, 0)
        evs[: len(ev), iter + 1] = ev
        # update lambda and stopping criterion
        fn1 = np.sum(ev[:c])
        fn2 = np.sum(ev[: c + 1])
        if fn1 > zr:
            lambda_param = 2 * lambda_param
        elif fn2 < zr:
            lambda_param = lambda_param / 2
            F = F_old
        else:
            print(f'iter = {iter + 1}  lambda: {lambda_param}')
            break
    # generating the clustering result (connected components)
    sU = (U + U.T) / 2
    graph = sparse.csr_matrix(sU)
    n_components, y = csgraph.connected_components(graph, directed=False)
    if n_components != c:
        print(f'Can not find the correct cluster number: {c}')
    y = y  # 0-based labels
    return y, U, S0, S0_initial, F, evs

def main_gmc_toy_examples():
    datadir = 'Dataset/'
    m = 2  # number of views
    dataname = input('Input the name of data set: (TwoMoon or ThreeRing)\n').strip()
    flag = False
    while True:
        if dataname == 'TwoMoon':
            dataf = os.path.join(datadir, dataname)
            c = 2
            mat = loadmat(dataf)
            X = [mat['X1'], mat['X2']] if 'X1' in mat else [mat['X'][0, 0], mat['X'][0, 1]]
            if isinstance(X[0], np.ndarray) and X[0].dtype == np.object_:
                X = [X[0].flat[0], X[1].flat[0]]
            y0 = mat['y0'] if 'y0' in mat else mat['Y']
            if y0.ndim > 1 and y0.shape[0] < y0.shape[1]:
                y0 = y0.T
            break
        elif dataname == 'ThreeRing':
            dataf = os.path.join(datadir, dataname)
            c = 3
            flag = True
            mat = loadmat(dataf)
            X = [mat['X1'], mat['X2']] if 'X1' in mat else [mat['X'][0, 0], mat['X'][0, 1]]
            if isinstance(X[0], np.ndarray) and X[0].dtype == np.object_:
                X = [X[0].flat[0], X[1].flat[0]]
            y0 = mat['y0'] if 'y0' in mat else mat['Y']
            if y0.ndim > 1 and y0.shape[0] < y0.shape[1]:
                y0 = y0.T
            break
        else:
            dataname = input('Please only input TwoMoon or ThreeRing\n').strip()
    # GMC expects each column = data point; toy data might be (n_samples, n_features)
    num = X[0].shape[0]
    data = [X[i].T for i in range(m)]
    predY, U, S0, S0_initial, F, evs = GMC(data, c, 1, 0)
    y0_flat = y0[:, 0] if y0.ndim > 1 else y0.ravel()
    nmi, ari, acc, pur = evaluate(y0_flat, predY)
    print(f'Data set {dataname} -> ACC:{acc:.4f}\tNMI:{nmi:.4f}\tARI:{ari:.4f}\tPurity:{pur:.4f}')
    marker_size = 20
    # Original data
    for v in range(m):
        lab = y0[:, v] if y0.ndim > 1 else y0.ravel()
        cLab = np.unique(lab)
        plt.figure()
        plt.plot(X[v][:, 0], X[v][:, 1], '.k', markersize=marker_size)
        plt.plot(X[v][lab == cLab[0], 0], X[v][lab == cLab[0], 1], '.r', markersize=20)
        plt.plot(X[v][lab == cLab[1], 0], X[v][lab == cLab[1], 1], '.', markersize=marker_size)
        if flag and len(cLab) > 2:
            plt.plot(X[v][lab == cLab[2], 0], X[v][lab == cLab[2], 1], '.', color=[79/255, 79/255, 79/255], markersize=marker_size)
        plt.axis('equal')
        plt.savefig(f'toy_original_view{v}.png')
        plt.close()
    # Original connected graph
    S1 = [S0_initial[v] for v in range(m)]
    for v in range(m):
        lab = y0[:, v] if y0.ndim > 1 else y0.ravel()
        cLab = np.unique(lab)
        plt.figure()
        plt.plot(X[v][:, 0], X[v][:, 1], '.k', markersize=marker_size)
        plt.plot(X[v][lab == cLab[0], 0], X[v][lab == cLab[0], 1], '.r', markersize=marker_size)
        plt.plot(X[v][lab == cLab[1], 0], X[v][lab == cLab[1], 1], '.', markersize=marker_size)
        if flag and len(cLab) > 2:
            plt.plot(X[v][lab == cLab[2], 0], X[v][lab == cLab[2], 1], '.', color=[79/255, 79/255, 79/255], markersize=marker_size)
        for ii in range(num):
            for jj in range(ii + 1):
                weight = S1[v][ii, jj]
                if weight > 0:
                    plt.plot([X[v][ii, 0], X[v][jj, 0]], [X[v][ii, 1], X[v][jj, 1]], '-', color=[0, 197/255, 205/255], linewidth=5*weight)
        plt.axis('equal')
        plt.savefig(f'toy_initial_graph_view{v}.png')
        plt.close()
    # Learned graph per view
    for v in range(m):
        lab = y0[:, v] if y0.ndim > 1 else y0.ravel()
        cLab = np.unique(lab)
        plt.figure()
        plt.plot(X[v][:, 0], X[v][:, 1], '.k', markersize=marker_size)
        plt.plot(X[v][lab == cLab[0], 0], X[v][lab == cLab[0], 1], '.r', markersize=marker_size)
        plt.plot(X[v][lab == cLab[1], 0], X[v][lab == cLab[1], 1], '.', markersize=marker_size)
        if flag and len(cLab) > 2:
            plt.plot(X[v][lab == cLab[2], 0], X[v][lab == cLab[2], 1], '.', color=[79/255, 79/255, 79/255], markersize=marker_size)
        for ii in range(num):
            for jj in range(ii + 1):
                weight = S0[v][ii, jj]
                if weight > 0:
                    plt.plot([X[v][ii, 0], X[v][jj, 0]], [X[v][ii, 1], X[v][jj, 1]], '-', color=[0, 197/255, 205/255], linewidth=5*weight)
        plt.axis('equal')
        plt.savefig(f'toy_learned_graph_view{v}.png')
        plt.close()
    # Learned unified graph
    U2 = U.copy()
    for v in range(m):
        lab = y0[:, v] if y0.ndim > 1 else y0.ravel()
        cLab = np.unique(lab)
        plt.figure()
        plt.plot(X[v][:, 0], X[v][:, 1], '.k', markersize=marker_size)
        plt.plot(X[v][lab == cLab[0], 0], X[v][lab == cLab[0], 1], '.r', markersize=marker_size)
        plt.plot(X[v][lab == cLab[1], 0], X[v][lab == cLab[1], 1], '.', markersize=marker_size)
        if flag and len(cLab) > 2:
            plt.plot(X[v][lab == cLab[2], 0], X[v][lab == cLab[2], 1], '.', color=[79/255, 79/255, 79/255], markersize=marker_size)
        for ii in range(num):
            for jj in range(ii + 1):
                weight = U2[ii, jj]
                if weight > 0:
                    plt.plot([X[v][ii, 0], X[v][jj, 0]], [X[v][ii, 1], X[v][jj, 1]], '-', color=[0, 197/255, 205/255], linewidth=5*weight)
        plt.axis('equal')
        plt.savefig(f'toy_unified_graph_view{v}.png')
        plt.close()
    print('Figures saved.')
    
def main_gmc_real_world():
    resultdir = 'Results/'
    os.makedirs(resultdir, exist_ok=True)
    dataname = ['100leaves', '3sources', 'BBC', 'BBCSport', 'HW', 'HW2sources', 'NGs', 'WebKB', 'Hdigit', 'Mfeat']
    runtimes = 1
    numdata = len(dataname)
    datadir = 'Dataset/'
    for cdata in range(numdata):
        idata = cdata
        dataf = os.path.join(datadir, dataname[idata])
        if not os.path.isfile(dataf + '.mat'):
            print(f'Skip {dataname[idata]}: file not found')
            continue
        mat = loadmat(dataf)
        # MATLAB: data, truelabel are in .mat
        X = mat['data']
        if isinstance(X, np.ndarray) and X.dtype == np.object_:
            # cell array: list of views
            X = [X.flat[i] for i in range(X.size)]
        else:
            X = [X]
        y0 = mat['truelabel'][:, 0]
        if y0.ndim > 1:
            y0 = y0.ravel()
        c = len(np.unique(y0))
        ACC = np.zeros(runtimes)
        NMI = np.zeros(runtimes)
        ARI = np.zeros(runtimes)
        for rtimes in range(runtimes):
            y, U, S0, S0_initial, F, evs = GMC(X, c)
            nmi, ari, acc, pur = evaluate(y0, y)
            ACC[rtimes] = acc
            NMI[rtimes] = nmi
            ARI[rtimes] = ari
            print(dataname[idata])
            print(f'=====In iteration {rtimes + 1}=====')
            print(f'ACC:{acc:.4f}\tNMI:{nmi:.4f}\tARI:{ari:.4f}\tPurity:{pur:.4f}')
        ncol = max(3, runtimes)
        Result = np.zeros((5, ncol))
        Result[0, :runtimes] = ACC
        Result[1, :runtimes] = NMI
        Result[2, :runtimes] = ARI
        Result[3, 0] = np.mean(ACC)
        Result[3, 1] = np.mean(NMI)
        Result[3, 2] = np.mean(ARI)
        Result[4, 0] = np.std(ACC)
        Result[4, 1] = np.std(NMI)
        Result[4, 2] = np.std(ARI)
        outpath = os.path.join(resultdir, f'{dataname[idata]}_result.mat')
        from scipy.io import savemat
        savemat(outpath, {'Result': Result, 'U': U, 'y0': y0, 'y': y})
        print(f'Saved {outpath}')

if __name__ == "__main__":
    """
    Graph-based Multi-view Clustering (GMC)
    min sum_v{ sum_i{||x_i - x_j||^2*s_ij + alpha*||s_i||^2} + w_v||U - Sv||^2 + lambda*trace(F'*Lu*F)}
    s.t Sv>=0, 1^T*Sv_i=1, U>=0, 1^T*Ui=1, F'*F=I
    Input:
    X: list of arrays, multi-view dataset; each element is a view, each column is a data point
    c: cluster number
    lambda: parameter (default 1)
    normData: whether to z-score normalize (default 1)
    Output:
    y: final clustering result (cluster indicator vector)
    U: learned unified matrix
    S0: similarity-induced graph (SIG) matrix for each view
    S0_initial: initial SIG matrices
    F: embedding representation
    evs: eigenvalues of learned graph Laplacian matrix
    """
    main_gmc_toy_examples()
    main_gmc_real_world()
    