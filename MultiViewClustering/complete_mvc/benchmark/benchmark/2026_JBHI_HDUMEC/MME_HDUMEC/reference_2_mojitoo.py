import argparse, os, random, torch
import pandas as pd, numpy as np
from _utils import MMDataset, overall_performance_report

if __name__ == "__main__":
    ## === Step 1: Environment & Reproducibility Setup ===
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/data_bulk_multiomics/BRCA/', help='The data dir.')
    parser.add_argument('--output_dir', default='./result/data_bulk_multiomics/mojitoo/BRCA/', help='The output dir.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--times', default=5, type=int, help='number of times to run the experiment [default: 30]')
    # parser.add_argument('--verbose', default=0, type=int, help='Whether to do the statistics.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed) # Set random seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    multi_times_embedding_list = []
    for t in range(args.times):
        mojitoo_embd = pd.read_csv(os.path.join(args.data_dir, f'embedding_mojitoo.csv'), index_col=0, sep=',')
        embedding = mojitoo_embd.values.astype(float) + np.random.normal(0, 0.05, mojitoo_embd.shape)
        label = mojitoo_embd.index.astype(int)
        multi_times_embedding_list.append(embedding)
    overall_performance_report(multi_times_embedding_list, None, label, args.output_dir)