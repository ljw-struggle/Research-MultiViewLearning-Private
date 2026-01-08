import random, numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


def myKNN(S, k, sigma=1.0):
    N = len(S)
    A = np.zeros((N,N))
    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours
        for j in neighbours_id: # xj is xi's neighbour
            A[i][j] = np.exp(-S[i][j]/2/sigma/sigma)
            A[j][i] = A[i][j] # mutually
    return A

def calLaplacianMatrix(adjacentMatrix):
    degreeMatrix = np.sum(adjacentMatrix, axis=1) # compute the degree matrix: D=sum(A)
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix # compute the Laplacian Matrix: L=D-A
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5))) # compute the square root of the degree matrix: D^(-1/2)
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix) # compute the Laplacian Matrix: L=D^(-1/2) * L * D^(-1/2)

def euclidDistance(x1, x2, sqrt_flag=False):
    res = np.sum((x1-x2)**2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res

def calEuclidDistanceMatrix(X):
    X = np.array(X)
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            S[i][j] = 1.0 * euclidDistance(X[i], X[j])
            S[j][i] = S[i][j]
    return S

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    # 1. Generate data which has two concentric circles
    data, label = datasets.make_circles(n_samples=500, factor=0.5, noise=0.05) # data: (500, 2), label: (500,)

    # 2. Construct the similarity matrix
    Similarity = calEuclidDistanceMatrix(data)
    Adjacent = myKNN(Similarity, k=10)

    # 3. Construct the Laplacian matrix
    Laplacian = calLaplacianMatrix(Adjacent)
    x, V = np.linalg.eig(Laplacian)
    x = zip(x, range(len(x)))
    x = sorted(x, key=lambda x:x[0])
    H = np.vstack([V[:,i] for (v, i) in x[:500]]).T
    sp_kmeans = KMeans(n_clusters=2).fit(H)
    pure_kmeans = KMeans(n_clusters=2).fit(data)

    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(data[:,0], data[:,1], s=10, color=colors[sp_kmeans.labels_])
    plt.title("Spectral Clustering")
    plt.subplot(122)
    plt.scatter(data[:,0], data[:,1], s=10, color=colors[pure_kmeans.labels_])
    plt.title("Kmeans Clustering")
    plt.savefig("spectral_clustering.png")