import os, sys, argparse, random, numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

try:
    from ..dataset import load_data
    from ..metric import evaluate
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from benchmark.dataset import load_data
    from benchmark.metric import evaluate

def benchmark_2011_JMLR_SKLEARN_AgglomerativeClustering(dataset_name='BDGP', 
                                                        linkage='ward', 
                                                        metric='euclidean', 
                                                        seed=42,
                                                        verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## 1. Set seed for reproducibility.
    random.seed(seed); 
    np.random.seed(seed); 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
    data, label, idx = next(iter(dataloader))
    data = torch.cat(data, dim=1).to(device) # shape: [data_size, sum(dims)]
    label = label.to(device) # shape: [data_size]
    
    ## 3. Run the clustering.
    data = data.cpu().numpy()
    label = label.cpu().numpy()
    model = AgglomerativeClustering(n_clusters=class_num, linkage=linkage, metric=metric)
    y_pred = model.fit_predict(data)
    nmi, ari, acc, pur = evaluate(label, y_pred)
    print("Clustering on concatenated views (AgglomerativeClustering):") if verbose else None
    print("ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}".format(acc, nmi, ari, pur)) if verbose else None
    return nmi, ari, acc, pur

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgglomerativeClustering")
    parser.add_argument("--dataset", default="BDGP", type=str)
    parser.add_argument("--linkage", default="ward", type=str, help="Linkage criterion")
    parser.add_argument("--metric", default="euclidean", type=str, help="Distance metric")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    nmi, ari, acc, pur = benchmark_2011_JMLR_SKLEARN_AgglomerativeClustering(
        dataset_name=args.dataset,
        linkage=args.linkage,
        metric=args.metric,
        seed=args.seed,
        verbose=False,
    )
    print("NMI = {:.4f}; ARI = {:.4f}; ACC = {:.4f}; PUR = {:.4f}".format(nmi, ari, acc, pur))
