
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy.io import loadmat
import numpy as np

import dhg

class Encoder(nn.Module):
    def __init__(self,num_views,input_dim,dim_high_feature,dim_proj_feature,num_layers=2):
        super().__init__()
        self.num_views = num_views
        self.num_layers = num_layers
        self.encoder = HyergraphEncoder(input_dim,dim_high_feature,self.num_layers) # shared HGNN
        self.projection = NodeProjectionHead(dim_high_feature,dim_proj_feature)     # shared projection head
        

    def forward(self, latent_data_views,hypergraph_views):
        node_encoder_output_features = list()
        node_proj_output_features = list()
        for idx in range(self.num_views):
            latent_data_view = latent_data_views[idx]
            hypergraph_view = hypergraph_views[idx]
            node_high_feature = self.encoder(latent_data_view,hypergraph_view)[0]
            node_proj_feature = self.projection(node_high_feature)

            node_encoder_output_features.append(node_high_feature)
            node_proj_output_features.append(node_proj_feature) 

        return node_encoder_output_features,node_proj_output_features
        


class ClusterHead(nn.Module):
    # cluster module for clustering
    def __init__(self,input_dim,hidden_dim, num_clusters):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一层隐藏层
        self.fc2 = nn.Linear(hidden_dim, num_clusters)  # 输出层，输出聚类概率
        
    def forward(self,z:Tensor):
        z = F.relu(self.fc1(z))  # 激活函数
        z = self.fc2(z)  # 输出层
        return F.softmax(z, dim=1)  # 应用 Softmax 函数，得到聚类概率分布 


# build HGNN+ model
# https://deephypergraph.readthedocs.io/en/0.9.4/tutorial/model.html#building-spatial-based-model
class HGNNPConv(nn.Module):
    # 超图卷积
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        drop_rate: float = 0.2,
       
    ):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
     
       
    def forward(self, X: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
        X = self.theta(X)
        Y = hg.v2e(X, aggr="mean")
        X_ = hg.e2v(Y, aggr="mean")
        X_ = self.drop(self.act(X_))
        return X_,Y



class NodeProjectionHead(nn.Module):
    # projection head for 对比学习
    def __init__(self,node_dim,proj_dim):
        super().__init__()
        self.node_dim = node_dim
        self.proj_dim = proj_dim
         # 参考的tricl projecthead过了两层dim又回去了
        self.fc1_n = nn.Linear(self.node_dim, self.proj_dim)
        self.fc2_n = nn.Linear(self.proj_dim, self.node_dim)

    def forward(self,z: Tensor):
       
        return self.fc2_n(F.elu(self.fc1_n(z)))


class ClusterHead(nn.Module):
    # cluster module for clustering
    def __init__(self,input_dim,hidden_dim, num_clusters):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一层隐藏层
        self.fc2 = nn.Linear(hidden_dim, num_clusters)  # 输出层，输出聚类概率
        
    def forward(self,z:Tensor):
        z = F.relu(self.fc1(z))  # 激活函数
        z = self.fc2(z)  # 输出层
        return F.softmax(z, dim=1)  # 应用 Softmax 函数，得到聚类概率分布 



class HyergraphEncoder(nn.Module):
    # stack hypergraph encoder layers
    def __init__(self,input_dim,output_dim,num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(HGNNPConv(self.input_dim,self.output_dim))

        if self.num_layers > 1:
            for i in range(1,self.num_layers):
                self.convs.append(HGNNPConv(self.output_dim,self.output_dim))
    
    def forward(self, node_input_feature:Tensor, hg):
        # input attribute matrix and hypergraph
        n, e = self.convs[0](node_input_feature,hg)
        if self.num_layers > 1:
            for i in range(1,self.num_layers):
                n, e = self.convs[i](n,hg)
        return n, e
    

# main network
class MVCnetwork(nn.Module):
    # def __init__(self,num_views, input_sizes, dims, dim_high_feature, dim_proj_feature, num_clusters):
    def __init__(self,num_views, input_sizes, dim_high_feature, dim_proj_feature, cluster_hidden_dim,num_clusters,num_layers=2):
        super().__init__()
        self.encoders = list()  # hypergraph encoders
        self.num_views = num_views
        self.num_layers = num_layers
        for idx in range(self.num_views):
            # 每个view创建hypergraph GNN
            self.encoders.append(HyergraphEncoder(input_sizes[idx],dim_high_feature,self.num_layers))   # gnn层
        self.encoders = nn.ModuleList(self.encoders)
        self.projection = NodeProjectionHead(dim_high_feature,dim_proj_feature) 
        self.cluster = ClusterHead(dim_high_feature,cluster_hidden_dim,num_clusters)
    
    def forward(self, data_views,hypergraph_views):
        node_encoder_output_features = list()
        node_proj_output_features = list()
        cluster_features = list()

        for idx in range(self.num_views ):
            data_view = data_views[idx]
            hypergraph_view = hypergraph_views[idx]
            # high_feature = self.encoders[idx](data_view,hypergraph_view)
            node_high_feature = self.encoders[idx](data_view,hypergraph_view)[0]
            node_proj_feature = self.projection(node_high_feature)
            cluster_feature = self.cluster(node_high_feature)

            node_encoder_output_features.append(node_high_feature)
            node_proj_output_features.append(node_proj_feature)
            cluster_features.append(cluster_feature)

        return node_encoder_output_features,node_proj_output_features,cluster_features

        # return lbps,features  #forward里输出contrastive loss里所需要的所有特征



if __name__ == "__main__":
    
    # clustermodel = ClusterHead(512,128,10)
    # z = torch.randn([500,512])
    # output = clustermodel(z)
    # print(output.shape)
    
    
    knn_num = 3
    dim_high_feature = 512

    # read data
    # path = "/root/0316/实验1/dataset/Caltech101-7.mat"
    path = "/root/0316/实验2/dataset/Wikipedia.mat"
    data = loadmat(path)
    X,y = data['X'], data['y']

    labels = y.squeeze(1)  # size (2866,)
    # print(labels.shape)
    num_views = X.shape[0]
    num_samples = labels.size
    num_clusters = np.unique(labels).size

    print(f"输入数据集的视图个数为:{num_views}")
    print(f"输入数据集的样本个数为：{num_samples}")
    print(f"数据集的聚类个数为：{num_clusters}")

    
    hypergraph_views = list()
    data_views = list()
    input_sizes = np.zeros(num_views, dtype=int)            # 每个view的输入的维度

    for idx in range(num_views):
        data_view = torch.from_numpy(X[idx][0].astype(np.float32)) 
        input_sizes[idx] = X[idx][0].shape[1]
        data_views.append(data_view)
        hypergraph_view = dhg.Hypergraph.from_feature_kNN(data_view, k=knn_num) 
        hypergraph_views.append(hypergraph_view)
    print(hypergraph_views)
    # print(data_views)
    print(f"输入数据集每个view的特征维度：{input_sizes}")       # [3183 3203]

    network = MVCnetwork(num_views, input_sizes, dim_high_feature=512, dim_proj_feature=128,cluster_hidden_dim=64,num_clusters=num_clusters)
    print(network)
    encoder_output_features,proj_output_features,cluster_output_features = network(data_views,hypergraph_views)


    for idx in range(num_views):
        print(f"view {idx} node representation shape:{encoder_output_features[idx].shape}, projectionhead output shape: {proj_output_features[idx].shape}, cluster_output_features shape:{cluster_output_features[idx].shape}")

    



 