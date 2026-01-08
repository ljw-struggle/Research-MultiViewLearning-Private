import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import (KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch)
from sklearn.mixture import GaussianMixture
from _utils import load_data, evaluate

# Define all datasets
DATASETS = [
    "BDGP",
    "MNIST-USPS",
    "CCV",
    "Fashion",
    "Caltech-2V",
    "Caltech-3V",
    "Caltech-4V",
    "Caltech-5V"
]

# Define all clustering algorithms and their parameter configuration function
def get_clustering_models(class_num, data_size):
    """Return a list of all clustering algorithm configurations, each element is (algorithm_name, model_instance)"""
    models = []
    # KMeans
    models.append(("KMeans", KMeans(n_clusters=class_num, init='k-means++', random_state=42, n_init=10)))
    # MiniBatchKMeans
    models.append(("MiniBatchKMeans", MiniBatchKMeans(n_clusters=class_num, random_state=42, n_init=10)))
    # # AffinityPropagation (does not require n_clusters)
    # models.append(("AffinityPropagation", AffinityPropagation(damping=0.6, random_state=42)))
    # # MeanShift (does not require n_clusters)
    # models.append(("MeanShift", MeanShift()))
    # SpectralClustering
    models.append(("SpectralClustering", SpectralClustering(n_clusters=class_num, random_state=42)))
    # AgglomerativeClustering
    models.append(("AgglomerativeClustering", AgglomerativeClustering(n_clusters=class_num)))
    # # DBSCAN (does not require n_clusters, but needs parameter tuning)
    # eps = 0.3 if data_size < 5000 else 0.5
    # min_samples = 10 if data_size < 5000 else 20
    # models.append(("DBSCAN", DBSCAN(eps=eps, min_samples=min_samples)))
    # # OPTICS (does not require n_clusters)
    # models.append(("OPTICS", OPTICS(eps=eps, min_samples=min_samples)))
    # Birch
    models.append(("Birch", Birch(n_clusters=class_num)))
    # GaussianMixture
    models.append(("GaussianMixture", GaussianMixture(n_components=class_num, covariance_type='full', random_state=42)))
    return models


if __name__ == "__main__":
    """Main function: run all clustering algorithms on all datasets and evaluate them"""
    results = {}
    for dataset_name in DATASETS:
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*80}")
        try:
            # Load dataset
            dataset, dims, view, data_size, class_num = load_data(dataset_name)
            dataloader = DataLoader(dataset, batch_size=data_size, shuffle=False)
            batch = next(iter(dataloader))
            multi_view_data = batch[0]  # Multi-view data list
            labels = batch[1].numpy()  # True labels
            X_concat = torch.cat(multi_view_data, dim=1).numpy()  # shape: (data_size, sum(dims))
            print(f"Dataset shape: {X_concat.shape}, Number of classes: {class_num}, Number of views: {view}")
            
            # Get all clustering algorithms
            clustering_models = get_clustering_models(class_num, data_size)
            dataset_results = {}
            
            # Run each clustering algorithm
            for model_name, model in clustering_models:
                try:
                    print(f"\n  Running {model_name}...")
                    # Train and predict
                    if hasattr(model, 'fit_predict'):
                        y_pred = model.fit_predict(X_concat)
                    else:
                        model.fit(X_concat)
                        y_pred = model.predict(X_concat)
                    
                    # # Handle noise points (DBSCAN and OPTICS may return -1)
                    # if -1 in y_pred:
                    #     # Assign noise points to the nearest cluster
                    #     unique_labels = np.unique(y_pred)
                    #     unique_labels = unique_labels[unique_labels != -1]
                    #     if len(unique_labels) > 0:
                    #         # Simple handling: replace -1 with 0 (or more complex processing can be done)
                    #         y_pred[y_pred == -1] = unique_labels[0]
                    
                    # Evaluate results
                    nmi, ari, acc, pur = evaluate(labels, y_pred)
                    dataset_results[model_name] = {'NMI': nmi, 'ARI': ari, 'ACC': acc, 'Purity': pur}
                    print(f"    NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}, Purity: {pur:.4f}")
                except Exception as e:
                    print(f"    Error running {model_name}: {str(e)}")
                    dataset_results[model_name] = {'Error': str(e)}
            results[dataset_name] = dataset_results
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {str(e)}")
            results[dataset_name] = {'Error': str(e)}
    
    # Print summary results
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}")
    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name}:")
        if 'Error' in dataset_results:
            print(f"  Error: {dataset_results['Error']}")
        else:
            for model_name, metrics in dataset_results.items():
                if 'Error' in metrics:
                    print(f"  {model_name}: Error - {metrics['Error']}")
                else:
                    print(f"  {model_name}: NMI={metrics['NMI']:.4f}, ARI={metrics['ARI']:.4f}, "
                          f"ACC={metrics['ACC']:.4f}, Purity={metrics['Purity']:.4f}")
