import os, csv, argparse, numpy as np
from benchmark.dataset import load_data

########################################################################################
### Benchmark of Traditional Clustering Algorithms
########################################################################################
from benchmark import benchmark_2011_AAAI_LSC
from benchmark import benchmark_2011_JMLR_SKLEARN_KMeans
from benchmark import benchmark_2011_JMLR_SKLEARN_MiniBatchKMeans
from benchmark import benchmark_2011_JMLR_SKLEARN_SpectralClustering
from benchmark import benchmark_2011_JMLR_SKLEARN_AgglomerativeClustering
from benchmark import benchmark_2011_JMLR_SKLEARN_Birch
from benchmark import benchmark_2011_JMLR_SKLEARN_GaussianMixture

def run_benchmark_2011_AAAI_LSC(dataset_name, use_view=-1, random_state_list=[1,2,3,4,5], output_dir='result/2011_AAAI_LSC/'):
    os.makedirs(output_dir, exist_ok=True)
    view_str = 'VIEW_ALL' if use_view == -1 else f'VIEW_{use_view}'
    csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
    nmi_list = []; ari_list = []; acc_list = []; pur_list = []
    for random_state in random_state_list:
        nmi, ari, acc, pur = benchmark_2011_AAAI_LSC(dataset_name=dataset_name, use_view=use_view, random_state=random_state)
        nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
        for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
            writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
        nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
        nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
        writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
        writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])
        
def run_benchmark_2011_JMLR_SKLEARN_KMeans(dataset_name, use_view=-1, random_state_list=[1,2,3,4,5], output_dir='result/2011_JMLR_SKLEARN_KMeans/'):
    os.makedirs(output_dir, exist_ok=True)
    view_str = 'VIEW_ALL' if use_view == -1 else f'VIEW_{use_view}'
    csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
    nmi_list = []; ari_list = []; acc_list = []; pur_list = []
    for random_state in random_state_list:
        nmi, ari, acc, pur = benchmark_2011_JMLR_SKLEARN_KMeans(dataset_name=dataset_name, use_view=use_view, random_state=random_state)
        nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
        for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
            writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
        nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
        nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
        writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
        writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])
        
def run_benchmark_2011_JMLR_SKLEARN_MiniBatchKMeans(dataset_name, use_view=-1, random_state_list=[1,2,3,4,5], output_dir='result/2011_JMLR_SKLEARN_MiniBatchKMeans/'):
    os.makedirs(output_dir, exist_ok=True)
    view_str = 'VIEW_ALL' if use_view == -1 else f'VIEW_{use_view}'
    csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
    nmi_list = []; ari_list = []; acc_list = []; pur_list = []
    for random_state in random_state_list:
        nmi, ari, acc, pur = benchmark_2011_JMLR_SKLEARN_MiniBatchKMeans(dataset_name=dataset_name, use_view=use_view, random_state=random_state)
        nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
        for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
            writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
        nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
        nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
        writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
        writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])
        
def run_benchmark_2011_JMLR_SKLEARN_SpectralClustering(dataset_name, use_view=-1, random_state_list=[1,2,3,4,5], output_dir='result/2011_JMLR_SKLEARN_SpectralClustering/'):
    os.makedirs(output_dir, exist_ok=True)
    view_str = 'VIEW_ALL' if use_view == -1 else f'VIEW_{use_view}'
    csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
    nmi_list = []; ari_list = []; acc_list = []; pur_list = []
    for random_state in random_state_list:
        nmi, ari, acc, pur = benchmark_2011_JMLR_SKLEARN_SpectralClustering(dataset_name=dataset_name, use_view=use_view, random_state=random_state)
        nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
        for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
            writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
        nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
        nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
        writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
        writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])
        
def run_benchmark_2011_JMLR_SKLEARN_AgglomerativeClustering(dataset_name, use_view=-1, random_state_list=[1,2,3,4,5], output_dir='result/2011_JMLR_SKLEARN_AgglomerativeClustering/'):
    os.makedirs(output_dir, exist_ok=True)
    view_str = 'VIEW_ALL' if use_view == -1 else f'VIEW_{use_view}'
    csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
    nmi_list = []; ari_list = []; acc_list = []; pur_list = []
    for random_state in random_state_list:
        nmi, ari, acc, pur = benchmark_2011_JMLR_SKLEARN_AgglomerativeClustering(dataset_name=dataset_name, use_view=use_view, random_state=random_state)
        nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
        for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
            writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
        nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
        nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
        writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
        writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])
        
def run_benchmark_2011_JMLR_SKLEARN_Birch(dataset_name, use_view=-1, random_state_list=[1,2,3,4,5], output_dir='result/2011_JMLR_SKLEARN_Birch/'):
    os.makedirs(output_dir, exist_ok=True)
    view_str = 'VIEW_ALL' if use_view == -1 else f'VIEW_{use_view}'
    csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
    nmi_list = []; ari_list = []; acc_list = []; pur_list = []
    for random_state in random_state_list:
        nmi, ari, acc, pur = benchmark_2011_JMLR_SKLEARN_Birch(dataset_name=dataset_name, use_view=use_view, random_state=random_state)
        nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
        for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
            writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
        nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
        nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
        writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
        writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])
        
def run_benchmark_2011_JMLR_SKLEARN_GaussianMixture(dataset_name, use_view=-1, random_state_list=[1,2,3,4,5], output_dir='result/2011_JMLR_SKLEARN_GaussianMixture/'):
    os.makedirs(output_dir, exist_ok=True)
    view_str = 'VIEW_ALL' if use_view == -1 else f'VIEW_{use_view}'
    csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
    nmi_list = []; ari_list = []; acc_list = []; pur_list = []
    for random_state in random_state_list:
        nmi, ari, acc, pur = benchmark_2011_JMLR_SKLEARN_GaussianMixture(dataset_name=dataset_name, use_view=use_view, random_state=random_state)
        nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
        for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
            writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
        nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
        nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
        writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
        writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])

# ########################################################################################
# ### Benchmark of Multi-View Clustering Algorithms based on AutoEncoder
# ########################################################################################
# from benchmark import benchmark_2022_CVPR_MFLVC
# from benchmark import benchmark_2023_ICCV_CVCL
# from benchmark import benchmark_2023_CVPR_GCFAgg
# from benchmark import benchmark_2024_CVPR_SCMVC

########################################################################################
### Benchmark of Multi-View Clustering Algorithms based on DEC (Deep Embedded Clustering)
########################################################################################
from benchmark import benchmark_2016_PMLR_DEC
from benchmark import benchmark_2017_IJCAI_IDEC
from benchmark import benchmark_2021_INSC_DEMVC
from benchmark import benchmark_2021_INSC_DEMVC_NC
from benchmark import benchmark_2022_TKDE_SDMVC
from benchmark import benchmark_2024_CVPR_MVCAN
from benchmark import benchmark_2026_JBHI_HDUMEC

def run_benchmark_2016_PMLR_DEC(dataset_name, use_view=-1, random_state_list=[1,2,3,4,5], output_dir='result/2016_PMLR_DEC/'):
    os.makedirs(output_dir, exist_ok=True)
    view_str = 'VIEW_ALL' if use_view == -1 else f'VIEW_{use_view}'
    csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
    nmi_list = []; ari_list = []; acc_list = []; pur_list = []
    for random_state in random_state_list:
        nmi, ari, acc, pur = benchmark_2016_PMLR_DEC(dataset_name=dataset_name, use_view=use_view, random_state=random_state)
        nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
        for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
            writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
        nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
        nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
        writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
        writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])
        
def run_benchmark_2017_IJCAI_IDEC(dataset_name, use_view=-1, random_state_list=[1,2,3,4,5], output_dir='result/2017_IJCAI_IDEC/'):
    os.makedirs(output_dir, exist_ok=True)
    view_str = 'VIEW_ALL' if use_view == -1 else f'VIEW_{use_view}'
    csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
    nmi_list = []; ari_list = []; acc_list = []; pur_list = []
    for random_state in random_state_list:
        nmi, ari, acc, pur = benchmark_2017_IJCAI_IDEC(dataset_name=dataset_name, use_view=use_view, random_state=random_state)
        nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
        for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
            writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
        nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
        nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
        writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
        writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])
        
def run_benchmark_2021_INSC_DEMVC(dataset_name, random_state_list=[1,2,3,4,5], output_dir='result/2021_INSC_DEMVC/'):
    os.makedirs(output_dir, exist_ok=True)
    view_str = 'VIEW_ALL'
    csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
    csv_path = os.path.join(output_dir, csv_filename)   
    assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
    nmi_list = []; ari_list = []; acc_list = []; pur_list = []
    for random_state in random_state_list:
        nmi, ari, acc, pur = benchmark_2021_INSC_DEMVC(dataset_name=dataset_name, random_state=random_state)
        nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
        for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
            writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
        nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
        nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
        writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
        writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])
        
def run_benchmark_2021_INSC_DEMVC_NC(dataset_name, random_state_list=[1,2,3,4,5], output_dir='result/2021_INSC_DEMVC_NC/'):
    os.makedirs(output_dir, exist_ok=True)
    view_str = 'VIEW_ALL'
    csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
    csv_path = os.path.join(output_dir, csv_filename)   
    assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
    nmi_list = []; ari_list = []; acc_list = []; pur_list = []
    for random_state in random_state_list:
        nmi, ari, acc, pur = benchmark_2021_INSC_DEMVC_NC(dataset_name=dataset_name, random_state=random_state)
        nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
        for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
            writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
        nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
        nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
        writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
        writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])
        
def run_benchmark_2022_TKDE_SDMVC(dataset_name, random_state_list=[1,2,3,4,5], output_dir='result/2022_TKDE_SDMVC/'):
    os.makedirs(output_dir, exist_ok=True)
    view_str = 'VIEW_ALL'
    csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
    nmi_list = []; ari_list = []; acc_list = []; pur_list = []
    for random_state in random_state_list:
        nmi, ari, acc, pur = benchmark_2022_TKDE_SDMVC(dataset_name=dataset_name, random_state=random_state)
        nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
        for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
            writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
        nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
        nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
        writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
        writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])
        
def run_benchmark_2024_CVPR_MVCAN(dataset_name, random_state_list=[1,2,3,4,5], output_dir='result/2024_CVPR_MVCAN/'):
    os.makedirs(output_dir, exist_ok=True)
    view_str = 'VIEW_ALL'
    csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
    nmi_list = []; ari_list = []; acc_list = []; pur_list = []
    for random_state in random_state_list:
        nmi, ari, acc, pur = benchmark_2024_CVPR_MVCAN(dataset_name=dataset_name, random_state=random_state)
        nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
        for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
            writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
        nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
        nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
        writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
        writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])   
        
def run_benchmark_2026_JBHI_HDUMEC(dataset_name, random_state_list=[1,2,3,4,5], output_dir='result/2026_JBHI_HDUMEC/'):
    os.makedirs(output_dir, exist_ok=True)
    view_str = 'VIEW_ALL'
    csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
    nmi_list = []; ari_list = []; acc_list = []; pur_list = []
    for random_state in random_state_list:
        nmi, ari, acc, pur = benchmark_2026_JBHI_HDUMEC(dataset_name=dataset_name, random_state=random_state)
        nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
        for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
            writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
        nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
        nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
        writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
        writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])
        
########################################################################################
### Benchmark of Multi-View Clustering Algorithms based on CL (Contrastive Learning)
########################################################################################
# from benchmark import benchmark_2022_CVPR_MFLVC
# from benchmark import benchmark_2023_ICCV_CVCL
# from benchmark import benchmark_2023_CVPR_GCFAgg
# from benchmark import benchmark_2023_MM_DealMVC
# from benchmark import benchmark_2024_TMM_SCMVC
# from benchmark import benchmark_2025_NIPS_SparseMVC
# from benchmark import benchmark_2025_TNNLS_MCMC

# def run_benchmark_2022_CVPR_MFLVC(dataset_name, use_view=-1, random_state_list=[1,2,3,4,5], output_dir='result/2022_CVPR_MFLVC/'):
#     os.makedirs(output_dir, exist_ok=True)
#     view_str = 'VIEW_ALL' if use_view == -1 else f'VIEW_{use_view}'
#     csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
#     csv_path = os.path.join(output_dir, csv_filename)
#     assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
#     nmi_list = []; ari_list = []; acc_list = []; pur_list = []
#     for random_state in random_state_list:

#         nmi, ari, acc, pur = benchmark_2022_CVPR_MFLVC(dataset_name=dataset_name, use_view=use_view, random_state=random_state)
#         nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
#     with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
#         for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
#             writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
#         nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
#         nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
#         writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
#         writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])
        
# def run_benchmark_2023_ICCV_CVCL(dataset_name, use_view=-1, random_state_list=[1,2,3,4,5], output_dir='result/2023_ICCV_CVCL/'):
#     os.makedirs(output_dir, exist_ok=True)
#     view_str = 'VIEW_ALL' if use_view == -1 else f'VIEW_{use_view}'
#     csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
#     csv_path = os.path.join(output_dir, csv_filename)
#     assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
#     nmi_list = []; ari_list = []; acc_list = []; pur_list = []
#     for random_state in random_state_list:
#         nmi, ari, acc, pur = benchmark_2023_ICCV_CVCL(dataset_name=dataset_name, use_view=use_view, random_state=random_state)
#         nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
#     with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
#         for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
#             writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
#         nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
#         nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
#         writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
#         writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])
        
# def run_benchmark_2023_CVPR_GCFAgg(dataset_name, use_view=-1, random_state_list=[1,2,3,4,5], output_dir='result/2023_CVPR_GCFAgg/'):
#     os.makedirs(output_dir, exist_ok=True)
#     view_str = 'VIEW_ALL' if use_view == -1 else f'VIEW_{use_view}'
#     csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
#     csv_path = os.path.join(output_dir, csv_filename)
#     assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
#     nmi_list = []; ari_list = []; acc_list = []; pur_list = []
#     for random_state in random_state_list:
#         nmi, ari, acc, pur = benchmark_2023_CVPR_GCFAgg(dataset_name=dataset_name, use_view=use_view, random_state=random_state)
#         nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
#     with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
#         for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
#             writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
#         nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
#         nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
#         writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
#         writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])
        
# def run_benchmark_2023_MM_DealMVC(dataset_name, use_view=-1, random_state_list=[1,2,3,4,5], output_dir='result/2023_MM_DealMVC/'):
#     os.makedirs(output_dir, exist_ok=True)
#     view_str = 'VIEW_ALL' if use_view == -1 else f'VIEW_{use_view}'
#     csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
#     csv_path = os.path.join(output_dir, csv_filename)
#     assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
#     nmi_list = []; ari_list = []; acc_list = []; pur_list = []  
#     for random_state in random_state_list:
#         nmi, ari, acc, pur = benchmark_2023_MM_DealMVC(dataset_name=dataset_name, use_view=use_view, random_state=random_state)
#         nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
#     with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
#         for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
#             writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
#         nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
#         nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
#         writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
#         writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])
        
# def run_benchmark_2024_TMM_SCMVC(dataset_name, use_view=-1, random_state_list=[1,2,3,4,5], output_dir='result/2024_TMM_SCMVC/'):
#     os.makedirs(output_dir, exist_ok=True)
#     view_str = 'VIEW_ALL' if use_view == -1 else f'VIEW_{use_view}'
#     csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
#     csv_path = os.path.join(output_dir, csv_filename)
#     assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
#     nmi_list = []; ari_list = []; acc_list = []; pur_list = []
#     for random_state in random_state_list:
#         nmi, ari, acc, pur = benchmark_2024_TMM_SCMVC(dataset_name=dataset_name, use_view=use_view, random_state=random_state)
#         nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
#     with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
#         for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
#             writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
#         nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
#         nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
#         writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
#         writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])
        
# def run_benchmark_2025_NIPS_SparseMVC(dataset_name, use_view=-1, random_state_list=[1,2,3,4,5], output_dir='result/2025_NIPS_SparseMVC/'):
#     os.makedirs(output_dir, exist_ok=True)
#     view_str = 'VIEW_ALL' if use_view == -1 else f'VIEW_{use_view}'
#     csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
#     csv_path = os.path.join(output_dir, csv_filename)
#     assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
#     nmi_list = []; ari_list = []; acc_list = []; pur_list = []
#     for random_state in random_state_list:
#         nmi, ari, acc, pur = benchmark_2025_NIPS_SparseMVC(dataset_name=dataset_name, use_view=use_view, random_state=random_state)
#         nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
#     with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
#         for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
#             writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
#         nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
#         nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
#         writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
#         writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])
        
# def run_benchmark_2025_TNNLS_MCMC(dataset_name, use_view=-1, random_state_list=[1,2,3,4,5], output_dir='result/2025_TNNLS_MCMC/'):
#     os.makedirs(output_dir, exist_ok=True)
#     view_str = 'VIEW_ALL' if use_view == -1 else f'VIEW_{use_view}'
#     csv_filename = f'{dataset_name.upper()}_{view_str.upper()}.csv'
#     csv_path = os.path.join(output_dir, csv_filename)
#     assert not os.path.exists(csv_path), f"File {csv_path} already exists, skipping..."
    
#     nmi_list = []; ari_list = []; acc_list = []; pur_list = []
#     for random_state in random_state_list:
#         nmi, ari, acc, pur = benchmark_2025_TNNLS_MCMC(dataset_name=dataset_name, use_view=use_view, random_state=random_state)
#         nmi_list.append(nmi); ari_list.append(ari); acc_list.append(acc); pur_list.append(pur)
    
#     with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Random_State', 'NMI', 'ARI', 'ACC', 'PUR'])
#         for random_state, nmi, ari, acc, pur in zip(random_state_list, nmi_list, ari_list, acc_list, pur_list):
#             writer.writerow([f'{random_state:04d}', f'{nmi:.4f}', f'{ari:.4f}', f'{acc:.4f}', f'{pur:.4f}'])
#         nmi_mean = np.mean(nmi_list); ari_mean = np.mean(ari_list); acc_mean = np.mean(acc_list); pur_mean = np.mean(pur_list)
#         nmi_std = np.std(nmi_list, ddof=0); ari_std = np.std(ari_list, ddof=0); acc_std = np.std(acc_list, ddof=0); pur_std = np.std(pur_list, ddof=0)
#         writer.writerow(['Mean', f'{nmi_mean:.4f}', f'{ari_mean:.4f}', f'{acc_mean:.4f}', f'{pur_mean:.4f}'])
#         writer.writerow(['Std ', f'{nmi_std:.4f}', f'{ari_std:.4f}', f'{acc_std:.4f}', f'{pur_std:.4f}'])
        
########################################################################################
### Main Function
########################################################################################

def generate_random_states(n=1, seed=42):
    np.random.seed(seed)
    return np.random.randint(0, 2**10, size=n).tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='BDGP')
    args = parser.parse_args()
    random_state_list = generate_random_states(n=1, seed=42)
    _, _, view, _, _ = load_data(args.dataset_name)
    
    ## Benchmark of Traditional Clustering Algorithms
    # view_list = [i for i in range(view)] + [-1]
    # for use_view in view_list:
    #     run_benchmark_2011_AAAI_LSC(dataset_name=args.dataset_name, use_view=use_view, random_state_list=random_state_list, output_dir=f'result/2011_AAAI_LSC/')
    #     run_benchmark_2011_JMLR_SKLEARN_KMeans(dataset_name=args.dataset_name, use_view=use_view, random_state_list=random_state_list, output_dir=f'result/2011_JMLR_SKLEARN_KMeans/')
    #     run_benchmark_2011_JMLR_SKLEARN_MiniBatchKMeans(dataset_name=args.dataset_name, use_view=use_view, random_state_list=random_state_list, output_dir=f'result/2011_JMLR_SKLEARN_MiniBatchKMeans/')
    #     run_benchmark_2011_JMLR_SKLEARN_SpectralClustering(dataset_name=args.dataset_name, use_view=use_view, random_state_list=random_state_list, output_dir=f'result/2011_JMLR_SKLEARN_SpectralClustering/')
    #     run_benchmark_2011_JMLR_SKLEARN_AgglomerativeClustering(dataset_name=args.dataset_name, use_view=use_view, random_state_list=random_state_list, output_dir=f'result/2011_JMLR_SKLEARN_AgglomerativeClustering/')
    #     run_benchmark_2011_JMLR_SKLEARN_Birch(dataset_name=args.dataset_name, use_view=use_view, random_state_list=random_state_list, output_dir=f'result/2011_JMLR_SKLEARN_Birch/')
    #     run_benchmark_2011_JMLR_SKLEARN_GaussianMixture(dataset_name=args.dataset_name, use_view=use_view, random_state_list=random_state_list, output_dir=f'result/2011_JMLR_SKLEARN_GaussianMixture/')

    ## Benchmark of Multi-View Clustering Algorithms based on DEC (Deep Embedded Clustering)
    # view_list = [i for i in range(view)] + [-1]
    # for use_view in view_list:
    #     run_benchmark_2016_PMLR_DEC(dataset_name=args.dataset_name, use_view=use_view, random_state_list=random_state_list, output_dir=f'result/2016_PMLR_DEC/')
    #     run_benchmark_2017_IJCAI_IDEC(dataset_name=args.dataset_name, use_view=use_view, random_state_list=random_state_list, output_dir=f'result/2017_IJCAI_IDEC/')
        
    # run_benchmark_2021_INSC_DEMVC(dataset_name=args.dataset_name, random_state_list=random_state_list, output_dir=f'result/2021_INSC_DEMVC/')
    # run_benchmark_2021_INSC_DEMVC_NC(dataset_name=args.dataset_name, random_state_list=random_state_list, output_dir=f'result/2021_INSC_DEMVC_NC/')
    # run_benchmark_2022_TKDE_SDMVC(dataset_name=args.dataset_name, random_state_list=random_state_list, output_dir=f'result/2022_TKDE_SDMVC/')
    # run_benchmark_2024_CVPR_MVCAN(dataset_name=args.dataset_name, random_state_list=random_state_list, output_dir=f'result/2024_CVPR_MVCAN/')
    run_benchmark_2026_JBHI_HDUMEC(dataset_name=args.dataset_name, random_state_list=random_state_list, output_dir=f'result/2026_JBHI_HDUMEC/')
    
    # ## Benchmark of Multi-View Clustering Algorithms based on CL (Contrastive Learning)
    # run_benchmark_2022_CVPR_MFLVC(dataset_name=args.dataset_name, use_view=use_view, random_state_list=random_state_list, output_dir=f'result/2022_CVPR_MFLVC/')
    # run_benchmark_2023_ICCV_CVCL(dataset_name=args.dataset_name, use_view=use_view, random_state_list=random_state_list, output_dir=f'result/2023_ICCV_CVCL/')
    # run_benchmark_2023_CVPR_GCFAgg(dataset_name=args.dataset_name, use_view=use_view, random_state_list=random_state_list, output_dir=f'result/2023_CVPR_GCFAgg/')
    # run_benchmark_2023_MM_DealMVC(dataset_name=args.dataset_name, use_view=use_view, random_state_list=random_state_list, output_dir=f'result/2023_MM_DealMVC/')
    # run_benchmark_2024_TMM_SCMVC(dataset_name=args.dataset_name, use_view=use_view, random_state_list=random_state_list, output_dir=f'result/2024_TMM_SCMVC/')
    # run_benchmark_2025_NIPS_SparseMVC(dataset_name=args.dataset_name, use_view=use_view, random_state_list=random_state_list, output_dir=f'result/2025_NIPS_SparseMVC/')
    # run_benchmark_2025_TNNLS_MCMC(dataset_name=args.dataset_name, use_view=use_view, random_state_list=random_state_list, output_dir=f'result/2025_TNNLS_MCMC/')
    