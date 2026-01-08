import os, h5py, parser, random, argparse
import pandas as pd, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients, Saliency
from main_train import AutoEncoder, AutoEncoder_Two_Modality, AutoEncoder_Three_Modality, setup_seed, read_label, read_data, preprocess_data, MMDataset, get_simulated_data_from_sampling, AverageMeter, accuracy

def test_model(model, dl, real_label, classify_dim=17, save_path = ''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_top_1_acc = AverageMeter('Acc@1', ':6.2f')
    each_celltype_top_1_acc = [AverageMeter('Acc@1', ':6.2f') for i in range(classify_dim)]; each_celltype_num = [0 for i in range(classify_dim)]
    model = model.eval(); classification_label = []; groundtruth_label = []; classification_prob = []
    with torch.no_grad():
        for i, batch_sample in enumerate(dl):
            x = torch.tensor(batch_sample['data'], dtype=torch.float32).reshape(x.size(0),-1)
            test_label = torch.tensor(batch_sample['label'], dtype=torch.long)
            x_recon, x_label, mu, var = model(x.to(device)); prob = torch.max(nn.Softmax()(x_label), 1)
            for j in range(x_recon.size(0)):
                classification_label.append(real_label[prob.indices[j]]); groundtruth_label.append(real_label[test_label[j]]); classification_prob.append(prob.values[j])
            test_top_1_acc.update(accuracy(x_label, test_label, topk=(1, ))[0], 1)
            for j in range(classify_dim):
                if len(test_label[test_label==j])!=0:
                    each_celltype_top_1_acc[j].update(accuracy(x_label[test_label==j,:], test_label[test_label==j], topk=(1, ))[0],1); 
                    each_celltype_num[j]=each_celltype_num[j] + len(test_label[test_label==j])
    for j in range(classify_dim):
        each_celltype_top_1_acc[j] = each_celltype_top_1_acc[j].avg
        print('cell type ID: ',j, '\t', '\t', 'cell type:', real_label[j], '\t', '\t', 'prec :', each_celltype_top_1_acc[j], 'number:', each_celltype_num[j], file = save_path)
    return model, each_celltype_top_1_acc, each_celltype_num, classified_label, groundtruth_label, prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Matilda')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--classification', type=bool, default= False, help='if augmentation or not')
    parser.add_argument('--query', type=bool, default= False, help='if the data is query of reference')
    parser.add_argument('--fs', type=bool, default= False, help='if doing feature selection or not')
    parser.add_argument('--fs_method', type=str, default= 'IntegratedGradient', help='choose the feature selection method')
    parser.add_argument('--dim_reduce', type=bool, default= False, help='save latent space')
    parser.add_argument('--simulation', type=bool, default= False, help='save simulation result')
    parser.add_argument('--simulation_ct', type=int, default= 1, help='simulate cell type')
    parser.add_argument('--simulation_num', type=int, default= 100, help='simulate cell number')
    parser.add_argument('--rna', metavar='DIR', default='NULL', help='path to train rna data')
    parser.add_argument('--adt', metavar='DIR', default='NULL', help='path to train adt data')
    parser.add_argument('--atac', metavar='DIR', default='NULL', help='path to train atac data')
    parser.add_argument('--cty', metavar='DIR', default='NULL', help='path to train cell type label')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--z_dim', type=int, default=100, help='the number of neurons in latent space')
    parser.add_argument('--hidden_rna', type=int, default=185, help='the number of neurons for RNA layer')
    parser.add_argument('--hidden_adt', type=int, default=30, help='the number of neurons for ADT layer')
    parser.add_argument('--hidden_atac', type=int, default=185, help='the number of neurons for ATAC layer')
    args = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.adt != 'NULL' and args.atac != 'NULL':
        rna_data = read_data(args.rna); adt_data = read_data(args.adt); atac_data = read_data(args.atac); label, label_namel_list = read_label(args.cty); classify_dim = len(label_namel_list)
        nfeatures_rna = rna_data.shape[1]; nfeatures_adt = adt_data.shape[1]; nfeatures_atac = atac_data.shape[1]; feature_num = nfeatures_rna + nfeatures_adt + nfeatures_atac
        mode = 'TEAseq'; data = torch.cat((preprocess_data(rna_data), preprocess_data(adt_data), preprocess_data(atac_data)), 1)   
        transformed_dataset = MMDataset(data, label); dl = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    if args.adt == 'NULL' and args.atac != 'NULL':
        rna_data = read_data(args.rna); atac_data = read_data(args.atac); label, label_namel_list = read_label(args.cty); classify_dim = len(label_namel_list)
        nfeatures_rna = rna_data.shape[1]; nfeatures_atac = atac_data.shape[1]; feature_num = nfeatures_rna  + nfeatures_atac
        mode = 'SHAREseq'; data = torch.cat((preprocess_data(rna_data), preprocess_data(atac_data)), 1)
        transformed_dataset = MMDataset(data, label); dl = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    if args.adt != 'NULL' and args.atac == 'NULL':
        rna_data = read_data(args.rna); adt_data = read_data(args.adt); label, label_namel_list = read_label(args.cty); classify_dim = len(label_namel_list)
        nfeatures_rna = rna_data.shape[1]; nfeatures_adt = adt_data.shape[1]; feature_num = nfeatures_rna + nfeatures_adt
        mode = 'CITEseq'; data = torch.cat((preprocess_data(rna_data), preprocess_data(adt_data)), 1)
        transformed_dataset = MMDataset(data, label); dl = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    if args.adt == 'NULL' and args.atac == 'NULL':
        rna_data = read_data(args.rna); label, label_namel_list = read_label(args.cty); classify_dim = len(label_namel_list)
        nfeatures_rna = rna_data.shape[1]; feature_num = nfeatures_rna
        mode = 'RNAonly'; data = preprocess_data(data)
        transformed_dataset = MMDataset(data, label); dl = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    print('The dataset is', mode)    
    path = 'query' if args.query else 'reference'
    model_save_path = '../trained_model/{}/'.format(mode); save_fs_eachcell = '../output/marker/{}/{}/'.format(mode,path)   
    rna_name = h5py.File(args.rna,'r')['matrix/features'][:]
    adt_name = h5py.File(args.adt,'r')['matrix/features'][:] if args.adt != 'NULL' else 'NULL'
    atac_name = h5py.File(args.atac,'r')['matrix/features'][:] if args.atac != 'NULL' else 'NULL'
    if mode == 'RNAonly': model = AutoEncoder(nfeatures_rna, args.hidden_rna, args.z_dim, classify_dim)
    if mode == 'CITEseq': model = AutoEncoder_Two_Modality(nfeatures_rna, nfeatures_adt, args.hidden_rna, args.hidden_adt, args.z_dim, classify_dim)
    if mode == 'SHAREseq': model = AutoEncoder_Two_Modality(nfeatures_rna, nfeatures_atac, args.hidden_rna, args.hidden_atac, args.z_dim, classify_dim)
    if mode == 'TEAseq': model = AutoEncoder_Three_Modality(nfeatures_rna, nfeatures_adt, nfeatures_atac, args.hidden_rna, args.hidden_adt, args.hidden_atac, args.z_dim, classify_dim)
    model = model.to(device) # model = nn.DataParallel(model).to(device) for multi gpu
    # Task 1: classification
    if args.classification == True:  
        os.makedirs('../output/classification/{}/{}'.format(mode,path), exist_ok=True)
        checkpoint = torch.load(os.path.join(model_save_path, 'model_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model, acc, num, classified_label, groundtruth_label, prob = test_model(model, dl, label_namel_list, classify_dim=classify_dim, save_path='../output/classification/{}/{}/accuracy_each_ct.txt'.format(mode, path))
        average = torch.mean(torch.Tensor(acc))
        for j in range(len(groundtruth_label)):
            print('cell ID: ', j, ' real cell type:', groundtruth_label[j], ' predicted cell type:', classified_label[j], ' probability:', round(float(prob[j]), 2), file = '../output/classification/{}/{}/accuracy_each_cell.txt'.format(mode, path))
    # Task 2: simulation 
    if args.simulation == True:
        print('simulate celltype index:', args.simulation_ct, ' cell type name:', label_namel_list[args.simulation_ct])
        os.makedirs('../output/simulation_result/{}/{}/'.format(mode,path), exist_ok=True)
        checkpoint = torch.load(os.path.join(model_save_path, 'simulation_model_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        index = (label == args.simulation_ct).nonzero(as_tuple=True)[0]
        aug_fold = int(args.simulation_num/int(index.size(0))); remaining_cell = int(args.simulation_num - aug_fold*int(index.size(0)))
        index = (label == args.simulation_ct).nonzero(as_tuple=True)[0]; anchor_data = data[index.tolist(),:]; anchor_label = label[index.tolist()]
        anchor_dl = DataLoader(MMDataset(anchor_data, anchor_label), batch_size=args.batch_size,shuffle=True, num_workers=0,drop_last=False)
        if aug_fold >= 1:
            reconstructed_data, reconstructed_label, real_data = get_simulated_data_from_sampling(model, anchor_dl)
            # For simulated data, we need to map the data to the same range as the real data and replace the nan value with the max value of the real data.
            reconstructed_data[reconstructed_data>torch.max(real_data)]=torch.max(real_data); reconstructed_data[reconstructed_data<torch.min(real_data)]=torch.min(real_data); reconstructed_data[torch.isnan(reconstructed_data)]=torch.max(real_data)
            new_data = reconstructed_data; new_label = reconstructed_label
            for i in range(aug_fold-1):
                reconstructed_data, reconstructed_label, real_data = get_simulated_data_from_sampling(model, anchor_dl)
                reconstructed_data[reconstructed_data>torch.max(real_data)]=torch.max(real_data); reconstructed_data[reconstructed_data<torch.min(real_data)]=torch.min(real_data); reconstructed_data[torch.isnan(reconstructed_data)]=torch.max(real_data)
                new_data = torch.cat((new_data,reconstructed_data),0); new_label = torch.cat((new_label,reconstructed_label.to(device)),0)
        reconstructed_data, reconstructed_label, real_data = get_simulated_data_from_sampling(model, anchor_dl)
        reconstructed_data[reconstructed_data>torch.max(real_data)]=torch.max(real_data); reconstructed_data[reconstructed_data<torch.min(real_data)]=torch.min(real_data); reconstructed_data[torch.isnan(reconstructed_data)]=torch.max(real_data)
        N = range(np.array(reconstructed_data.size(0))); ds_index = random.sample(N, remaining_cell); reconstructed_data = reconstructed_data[ds_index,:]; reconstructed_label = reconstructed_label[ds_index]
        new_data = reconstructed_data if aug_fold ==0 else torch.cat((new_data, reconstructed_data),0)
        new_label = reconstructed_label if aug_fold ==0 else torch.cat((new_label, reconstructed_label),0)
        index = (label != args.simulation_ct).nonzero(as_tuple=True)[0]; non_anchor_data = data[index.tolist(),:]; non_anchor_label = label[index.tolist()]; real_data = data; real_label = label
        sim_data = torch.cat((non_anchor_data,new_data),0); sim_label = torch.cat((non_anchor_label,new_label.to(device)),0) # sim_data includes the simulated data and the real data with the other cell types
        sim_data_rna = sim_data[:, 0:nfeatures_rna]; real_data_rna = real_data[:, 0:nfeatures_rna]   
        sim_data_adt = sim_data[:, nfeatures_rna:(nfeatures_rna+nfeatures_adt)]; real_data_adt = real_data[:, nfeatures_rna:(nfeatures_rna+nfeatures_adt)]
        sim_data_atac = sim_data[:, (nfeatures_rna+nfeatures_adt):(nfeatures_rna+nfeatures_adt+nfeatures_atac)]; real_data_atac = real_data[:, (nfeatures_rna+nfeatures_adt):(nfeatures_rna+nfeatures_adt+nfeatures_atac)]
        cell_name_real = ['cell_{}'.format(b) for b in range(0, real_data_rna.size(0))]; cell_name_sim = ['cell_{}'.format(b) for b in range(0, sim_data_rna.size(0))]
        sim_label_new = [label_namel_list[sim_label[j]] for j in range(sim_data_rna.size(0))]; real_label_new = [label_namel_list[real_label[j]] for j in range(real_data_rna.size(0))]
        rna_name_new = [str(rna_name[i], encoding='utf-8') for i in range(sim_data_rna.size(1))]    
        adt_name_new = [str(adt_name[i], encoding='utf-8') for i in range(sim_data_adt.size(1))] if mode == 'CITEseq' or mode == 'TEAseq' else []
        atac_name_new = [str(atac_name[i], encoding='utf-8') for i in range(sim_data_atac.size(1))] if mode == 'SHAREseq' or mode == 'TEAseq' else []     
        pd.DataFrame(sim_data_adt.cpu().numpy(), index = cell_name_sim, columns = adt_name_new).to_csv( '../output/simulation_result/{}/{}/sim_data_adt.csv'.format(mode,path)) if mode == 'CITEseq' or mode == 'TEAseq' else None
        pd.DataFrame(real_data_adt.cpu().numpy(), index = cell_name_real, columns = adt_name_new).to_csv( '../output/simulation_result/{}/{}/real_data_adt.csv'.format(mode,path)) if mode == 'CITEseq' or mode == 'TEAseq' else None
        pd.DataFrame(sim_data_atac.cpu().numpy(), index = cell_name_sim, columns = atac_name_new).to_csv( '../output/simulation_result/{}/{}/sim_data_atac.csv'.format(mode,path)) if mode == 'SHAREseq' or mode == 'TEAseq' else None
        pd.DataFrame(real_data_atac.cpu().numpy(), index = cell_name_real, columns = atac_name_new).to_csv( '../output/simulation_result/{}/{}/real_data_atac.csv'.format(mode,path)) if mode == 'SHAREseq' or mode == 'TEAseq' else None
        pd.DataFrame(sim_data_rna.cpu().numpy(), index = cell_name_sim, columns = rna_name_new).to_csv( '../output/simulation_result/{}/{}/sim_data_rna.csv'.format(mode,path))
        pd.DataFrame(real_data_rna.cpu().numpy(), index = cell_name_real, columns = rna_name_new).to_csv( '../output/simulation_result/{}/{}/real_data_rna.csv'.format(mode,path))
        pd.DataFrame(sim_label_new, index = cell_name_sim, columns = ['label']).to_csv( '../output/simulation_result/{}/{}/sim_label.csv'.format(mode,path))
        pd.DataFrame(real_label_new, index = cell_name_real, columns = ['label']).to_csv( '../output/simulation_result/{}/{}/real_label.csv'.format(mode,path))
    # Task 3: dimension reduction
    if args.dim_reduce == True:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(os.path.join(model_save_path, 'model_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval()
        data = []; label = []; encodings = []
        with torch.no_grad():
            for i, batch_sample in enumerate(dl):
                x = torch.tensor(batch_sample['data'], dtype=torch.float32).reshape(x.size(0),-1); y = torch.tensor(batch_sample['label'], dtype=torch.long)
                encoding = model.encoder(x.to(device)); data.append(x); label.append(y); encodings.append(encoding.cpu().numpy())
        simulated_data_ls = torch.cat(encodings, dim=0); data_ls = torch.cat(data, dim=0); label_ls = torch.cat(label, dim=0)
        # For simulated data, we need to map the data to the same range as the real data and replace the nan value with the max value of the real data.
        simulated_data_ls[simulated_data_ls>torch.max(data)]=torch.max(data_ls); simulated_data_ls[simulated_data_ls<torch.min(data)]=torch.min(data_ls); simulated_data_ls[torch.isnan(simulated_data_ls)]=torch.max(data_ls)
        os.makedirs('../output/dim_reduce/{}/{}/'.format(mode,path), exist_ok=True)
        feature_index = ['feature_{}'.format(b) for b in range(0, simulated_data_ls.size(1))]   
        cell_name_real = ['cell_{}'.format(b) for b in range(0, data.size(0))]  
        real_label_new = [label_namel_list[label[j]] for j in range(data.size(0))]
        pd.DataFrame(simulated_data_ls.cpu().numpy(), index = cell_name_real, columns = feature_index).to_csv( '../output/dim_reduce/{}/{}/latent_space.csv'.format(mode,path))
        pd.DataFrame(real_label_new, index = cell_name_real, columns = ['label']).to_csv('../output/dim_reduce/{}/{}/latent_space_label.csv'.format(mode,path))
    # Task 4: feature selection
    if args.fs == True:
        rna_name_new = [str(rna_name[i], encoding='utf-8') for i in range(nfeatures_rna)]
        adt_name_new = [str(adt_name[i], encoding='utf-8') for i in range(nfeatures_adt)] if mode == 'CITEseq' or mode == 'TEAseq' else []
        atac_name_new = [str(atac_name[i], encoding='utf-8') for i in range(nfeatures_atac)] if mode == 'SHAREseq' or mode == 'TEAseq' else []
        features = rna_name_new + adt_name_new + atac_name_new
        checkpoint = torch.load(os.path.join(model_save_path, 'model_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        classify_model = nn.Sequential(*list(model.children()))[0:2]
        deconv = Saliency(classify_model) if args.fs_method == 'Saliency' else IntegratedGradients(classify_model) # TODO: bug here
        for i in range(classify_dim):
            train_index_fs = np.array([t.cpu().numpy() for t in torch.where(label==i)])
            train_data_each_celltype_fs = data[train_index_fs,:].reshape(-1,feature_num)
            attribution = torch.zeros(1,feature_num)
            for j in range(train_data_each_celltype_fs.size(0)-1):
                attribution = attribution.to(device) + torch.abs(deconv.attribute(train_data_each_celltype_fs[j:j+1,:], target=i))
            attribution_mean = torch.mean(attribution,dim=0)
            pd.DataFrame(attribution_mean.cpu().numpy(), index = features, columns = ['importance score']).to_csv(save_fs_eachcell+'/fs.'+'celltype'+str(i)+'.csv')