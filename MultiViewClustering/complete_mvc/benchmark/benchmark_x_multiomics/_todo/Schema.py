import schema
# from schema import SchemaQP
import numpy as np
import sklearn
# dataset = 'sim1'
# dataset = 'sim2'
dataset = 'sim3_dropout'
rna_path = './../datasets/new_simulation_dataset/{}/rna.npy'.format(dataset)
protein_path = './../datasets/new_simulation_dataset/{}/adt.npy'.format(dataset)
atac_path = './../datasets/new_simulation_dataset/{}/atac.npy'.format(dataset)
# dataset = 'tea'
# dataset = 'dogma'
# dataset = 'neat'
# rna_path = './my_{}_data/rna_3000hvg_pca.npy'.format(dataset)
# protein_path = './my_{}_data/adt.npy'.format(dataset)
# atac_path = './my_{}_data/atac.npy'.format(dataset)

# rna_path = './my_{}_data/fts_rna.npy'.format(dataset)
# protein_path = './my_{}_data/fts_protein.npy'.format(dataset)
# atac_path = './my_{}_data/fts_atac.npy'.format(dataset)
rna_data = np.load(rna_path)
protein_data = np.load(protein_path)
atac_data = np.load(atac_path)
print(rna_data.shape)
print(protein_data.shape)
print(atac_data.shape)
sqp99 = schema.SchemaQP(0.99, mode='affine', params= {"decomposition_model":"pca", "num_top_components":50, "do_whiten": 0, "dist_npairs": 5000000})
rna_protein = sqp99.fit_transform(rna_data, [protein_data], ['feature_vector'], [1])
rna_atac = sqp99.fit_transform(rna_data, [atac_data], ['feature_vector'], [1])
# rna_protein = np.load('rna_protein.npy')
# rna_atac = np.load('rna_atac.npy')
schema_embedding = np.hstack((rna_protein,rna_atac))
print(rna_protein.shape)
print(rna_atac.shape)
print(schema_embedding.shape)
# np.save('rna_protein_{}.npy'.format(dataset),rna_protein)
# np.save('rna_atac_{}.npy'.format(dataset),rna_atac)
np.save('schema_embedding_{}.npy'.format(dataset),schema_embedding)

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

if __name__ == '__main__':
    model = MLP(10, 5, 2)
    print(model)