import os, torch, random, numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics.cluster import contingency_matrix, normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from scipy.optimize import linear_sum_assignment
from torchvision import datasets, transforms
from model import SpectralNet

# def get_eigenvalues_eigenvectors(A: np.ndarray) -> np.ndarray:
#     # Computes the eigenvalues and eigenvectors of a given matrix A and sorts them in increasing order of the eigenvalues.
#     vecs, vals, _ = np.linalg.svd(A)
#     sorted_vals = vals[np.argsort(vals)] # sort the eigenvalues in increasing order
#     sorted_vecs = vecs[:, np.argsort(vals)] # sort the eigenvectors by the eigenvalues
#     return sorted_vals, sorted_vecs

# def get_grassman_distance(A: np.ndarray, B: np.ndarray) -> float:
#     # Computes the Grassmann distance between the subspaces spanned by the columns of A and B.
#     M = np.dot(np.transpose(A), B)
#     _, s, _ = np.linalg.svd(M, full_matrices=False)
#     s = 1 - np.square(s)
#     grassmann = np.sum(s)
#     return grassmann

def get_hungarian_alignment(y_true, y_pred): # y_pred and y_true are numpy arrays, same shape
    D = int(max(np.max(y_true), np.max(y_pred)) + 1); M = confusion_matrix(y_true, y_pred, labels=np.arange(D))  # shape: (num_true_clusters, num_pred_clusters) 
    cost_matrix = M.max() - M # shape: [num_true_clusters, num_pred_clusters]
    ind = linear_sum_assignment(cost_matrix) # align clusters using the Hungarian algorithm, ind[0] is the true clusters, ind[1] is the predicted clusters
    matched_dict = {i: j for i, j in zip(ind[1], ind[0])} # shape: {pred_cluster: true_cluster}
    matched_matrix = np.zeros((D, D)); matched_matrix[ind[1], ind[0]] = 1 # shape: [num_pred_clusters, num_true_clusters]
    return matched_dict, matched_matrix # matched_dict: {pred_cluster: true_cluster}, matched_matrix: (num_pred_clusters, num_true_clusters)

def evaluate(label, pred):
    matched_dict, _ = get_hungarian_alignment(label, pred)
    pred_aligned = np.array([matched_dict[i] for i in pred])
    print(confusion_matrix(label, pred_aligned))
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    pur = np.sum(np.amax(contingency_matrix(label, pred), axis=0)) / len(label)
    acc = accuracy_score(label, pred_aligned)
    # f1_weighted = f1_score(label, pred_aligned, average="weighted")
    # f1_macro = f1_score(label, pred_aligned, average="macro")
    # f1_micro = f1_score(label, pred_aligned, average="micro")
    # asw = silhouette_score(embedding, pred_aligned) # silhouette score
    return nmi, ari, acc, pur

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    os.makedirs("./result", exist_ok=True)
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    x_train, y_train = zip(*train_set); x_train, y_train = torch.cat(x_train), torch.Tensor(y_train)
    x_test, y_test = zip(*test_set); x_test, y_test = torch.cat(x_test), torch.Tensor(y_test)
    data = torch.cat([x_train, x_test]); data = data.view(data.size(0), -1); y = torch.cat([y_train, y_test])
    spectralnet = SpectralNet(n_clusters=10, should_use_ae=True, should_use_siamese=True)
    spectralnet.fit(data, y)
    cluster_assignments, embeddings = spectralnet.predict(data)
    spectralnet.visualize_embedding(data, embeddings, y.detach().cpu().numpy(), save_path="./result/")
    nmi, ari, acc, pur = evaluate(y.detach().cpu().numpy(), cluster_assignments)
    print(f"NMI: {np.round(nmi, 3)}"); print(f"ARI: {np.round(ari, 3)}"); print(f"ACC: {np.round(acc, 3)}"); print(f"PUR: {np.round(pur, 3)}")
    