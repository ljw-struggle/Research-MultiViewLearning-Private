import numpy as np
import h5py
import pandas as pd

def load_tsv(file):
    df = pd.read_csv(file, sep='\t', header= 0, index_col= 0)
    return df

def load_excel(file):
    df = pd.read_excel(file,header=0)
    return df

def load_csv(file):
    df = pd.read_csv(file, sep=',', header= 0)
    return df

def return_key(dict,val):
    for key, value in dict.items():
        if val in value:
            return key
    return('Key Not Found')

def get_id_from_antigen(proteins, df_antigen):
    data = []
    delete = []
    antigens = np.array(df_antigen['antigen_name'])
    for protein in proteins:
        if protein in antigens:
            index = df_antigen[df_antigen.antigen_name == protein].index.tolist()[0]
            id = df_antigen['Entry'][index]
            data.append([protein,id])
        else:
            delete.append(protein)
    print(str(len(delete))+' protein sequences not found in antigen.')
    return data, delete

def get_id_from_annotation(data, delete, df_annotation):
    names = np.array(df_annotation['Protein_name_prime'])
    
    #generate protein-gene names dict
    dict_genes = {}
    for i in list(df_annotation.index):
        if str(df_annotation['Gene Names'][i]) != 'nan':
            dict_genes[df_annotation['Protein_name_prime'][i]] = df_annotation['Gene Names'][i].split(' ')
    genes = sum(dict_genes.values(), [])
    print(str(len(genes)) +' genes in annotation.')

    old_delete = delete.copy()
    for protein in old_delete:
        # scan Protein_name_prime
        if protein in names:
            index = df_annotation[df_annotation.Protein_name_prime == protein].index.tolist()[0]
            id = df_annotation['Entry'][index]
            data.append([protein,id])
            delete.remove(protein)
        # scan Gene Names
        elif protein in genes:
            name = return_key(dict_genes,protein)
            index = df_annotation[df_annotation.Protein_name_prime == name].index.tolist()[0]
            id = df_annotation['Entry'][index]
            data.append([protein,id])
            delete.remove(protein)
    print(str(len(delete))+' protein sequences not found in annotation.')
    return data, delete




file_emb = 'E:/Papers/20230606-OmicsTrans/Protein/ProtT5-human-embedding/per-protein.h5'
file_protein = 'E:/Papers/20230606-OmicsTrans/Datasets/RNA-Protein/REAP-PBMC/GSM2685244_protein_3_PBMCs_matrix.txt'
file_antigen = 'E:/Papers/20230606-OmicsTrans/Protein/uniprot_protein_antigen_information.csv'
file_seq = 'E:/Papers/20230606-OmicsTrans/Protein/uniprot_protein_information_human.xlsx'
file_out = 'E:/Papers/20230606-OmicsTrans/Datasets/RNA-Protein/REAP-PBMC/ProtT5_embedding.npz'

df_protein = load_tsv(file_protein)
df_antigen = load_csv(file_antigen)
df_seq = load_excel(file_seq)

proteins = np.array(df_protein.index)
#process protein names
for i in range(len(proteins)):
    proteins[i] = proteins[i].split('_')[0]

#find protein from antigen
data, delete = get_id_from_antigen(proteins, df_antigen)
#find protein from protein and gene
data, delete = get_id_from_annotation(data, delete,  df_seq)
print(str(len(data)) +' /'+str(len(proteins)) +' left based on sequences.')
print('The entry of the following proteins can not be found.')
print(delete)

emb_name = []
emb = []
with h5py.File(file_emb, "r") as file:
    print(f"Number of entries in ProtT5 file: {len(file.items())}")
    for i in range(len(data)):
        if data[i][1] in file.keys():
            emb_name.append(data[i][0])
            emb.append(np.array(file[data[i][1]]))
        else:
            delete.append(data[i][0])
    emb = np.array(emb)

    print('The ProtT5 of the following proteins can not be found.')
    print(delete)
    print('We got '+ str(len(emb_name))+' proteins with 1024-dimentional embedding.')
    np.savez(file_out, protein = emb_name, embedding = emb)


# import h5py
# import numpy as np

# with h5py.File("per-protein.h5", "r") as file:
#     print(f"number of entries: {len(file.items())}")
#     for sequence_id, embedding in file.items():
#         print(f"id: {sequence_id}, embeddings shape: {embedding.shape}, embeddings mean: {np.array(embedding).mean()}")