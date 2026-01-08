import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

from scipy.io import loadmat
import dhg
import numpy as np

# 参考：https://github.com/wooner49/TriCL/blob/main/TriCL/models.py



class MultiContrastiveLoss(nn.Module):
    def __init__(self,temperature, batch_size):
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.loss_func1 = ContrastiveLoss()
        self.loss_func2 = ContrastiveLoss()
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        

    
    def forward(self,z1,z2,h1,h2,para1:0.5,para2:0.5,m1,m2):
        m1 = m1.detach()  # 确保 m1 不参与梯度计算
        m2 = m2.detach()  # 确保 m2 不参与梯度计算
        loss1 = self.loss_func1.batch_loss(z1,z2,self.temperature ,self.batch_size,m1,m2)
        loss2 = self.loss_func2.batch_loss(h1,h2,self.temperature ,self.batch_size,m2,m1)

        KL_criterion = torch.nn.KLDivLoss(size_average=False)

        # kl1 = KL_criterion(h1.log(), h2)
        # print(kl1)
        return para1*loss1 + para2*loss2 
         
    def compute_contrastive_loss(self,h1,h2,hg1,hg2,tau:float,batch_size:int,if_aug=True):
        # cross view contrastive loss
        num_samples = h1.size(0)
        indices = torch.arange(0, num_samples)
        num_batches = (num_samples-1)//batch_size + 1
        losses = []

        hg_incdence_matrix =  hg1.H.to_dense()

        for i in range(num_batches):
            mask = indices[i*batch_size:(i+1)*batch_size]
            hyperedge_mask = hg_incdence_matrix[i*batch_size:(i+1)*batch_size]
            batch_loss = self.__semi_loss_batch(h1[mask],h2[mask],tau)
            # batch_loss = self.__semi_intra_hyperedge_negative_loss(h1[mask],h2[mask],hyperedge_mask,tau)
            losses.append(batch_loss)
        

        # 按样本量加权平均（尤其当最后一个batch不足batch_size时）
        total_loss = torch.cat(losses).mean()
        return total_loss
    def cosine_similarity(self, z1: torch.Tensor, z2: torch.Tensor):
        print(z1.shape)
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    
    def f(self, x, tau):
        return torch.exp(x / tau)



    def __semi_loss_batch(self,h1:torch.Tensor,h2:torch.Tensor,tau:float):
        # print(f"h1 shape{h1.shape},h2 shape{h2.shape}")
        between_sim = self.f(self.cosine_similarity(h1, h2), tau)
        pos_sim = between_sim.diag()  # 对角线
        neg_sim = between_sim.sum(1)  # 沿维度1求和
        # print(pos_sim.shape,neg_sim.shape)
        batch_loss = -torch.log(pos_sim / (pos_sim + neg_sim))
        # batch_loss = -torch.log(pos_sim / (neg_sim))

        return batch_loss
    

    def __semi_loss_batch_with_mask(self,h1:torch.Tensor,h2:torch.Tensor,mask,tau:float):
        # print(f"h1 shape{h1.shape},h2 shape{h2.shape}")
        between_sim = self.f(self.cosine_similarity(h1, h2), tau)
        pos_sim = between_sim.diag()  # 对角线
        neg_sim = torch.mm(mask,between_sim).sum(1)  # 沿维度1求和  ，只保留负样本
        # print(pos_sim.shape,neg_sim.shape)
        batch_loss = -torch.log(pos_sim / (pos_sim + neg_sim))
        # batch_loss = -torch.log(pos_sim / (neg_sim))

        return batch_loss
    
    def compute_inner_view_loss(self,h1:torch.Tensor,h2:torch.Tensor,hg1,hg2,tau:float,batch_size):
        # cross view contrastive loss
        num_samples = h1.size(0)
        indices = torch.arange(0, num_samples)
        num_batches = (num_samples-1)//batch_size + 1
        losses = []

        hg1_incdence_matrix =  hg1.H.to_dense()
        hg2_incdence_matrix =  hg2.H.to_dense()


        for i in range(num_batches):
            mask = indices[i*batch_size:(i+1)*batch_size]
            hyperedge_mask1 = hg1_incdence_matrix[i*batch_size:(i+1)*batch_size]
            negative_mask1 = torch.matmul(hyperedge_mask1,hyperedge_mask1.T)
            negative_mask1[negative_mask1 >= 1] = 1

            hyperedge_mask2 = hg2_incdence_matrix[i*batch_size:(i+1)*batch_size]
            negative_mask2 = torch.matmul(hyperedge_mask2,hyperedge_mask2.T)
            negative_mask2[negative_mask2 >= 1] = 1

            batch_loss_1 = self.__semi_loss_batch_with_mask(h1[mask],h1[mask],mask=negative_mask1,tau=tau)
            batch_loss_1_2 = self.__semi_loss_batch_with_mask(h1[mask],h2[mask], mask=negative_mask2,tau=tau)
            losses.append(batch_loss_1)
            losses.append(batch_loss_1_2)

        

        # 按样本量加权平均（尤其当最后一个batch不足batch_size时）
        total_loss = torch.cat(losses).mean()
        return total_loss
        return loss




    def compute_inner_view_loss2(self,h1:torch.Tensor,hg1,tau:float,batch_size):
        # cross view contrastive loss
        num_samples = h1.size(0)
        indices = torch.arange(0, num_samples)
        num_batches = (num_samples-1)//batch_size + 1
        losses = []

        hg1_incdence_matrix =  hg1.H.to_dense()


        for i in range(num_batches):
            mask = indices[i*batch_size:(i+1)*batch_size]
            hyperedge_mask1 = hg1_incdence_matrix[i*batch_size:(i+1)*batch_size]
            negative_mask1 = torch.matmul(hyperedge_mask1,hyperedge_mask1.T)
            negative_mask1[negative_mask1 > 1] = 1


            batch_loss_1 = self.__semi_loss_batch_with_mask(h1[mask],h1[mask],mask=negative_mask1,tau=tau)
            losses.append(batch_loss_1)
      

        # 按样本量加权平均（尤其当最后一个batch不足batch_size时）
        total_loss = torch.cat(losses).mean()
        return total_loss
    

# compute_inner_view_loss(old_proj_output_features[0],proj_output_features[0],hypergraph_views[0],hypergraph_views_copy[0],tau=0.5,batch_size=600)

class ContrastiveLoss(nn.Module):
    def __init__(self):
          super().__init__()
    
        
    def batch_loss(self,z1:Tensor, z2: Tensor,temperature: float, batch_size: int,m1,m2):
        l1 = self.semi_loss_batch(z1,z2,temperature,batch_size,m1)
        l2 = self.semi_loss_batch(z2,z1,temperature,batch_size,m2)

        loss = (l1+l2) * 0.5
        loss = loss.mean() 
        return loss

    def f(self, x, tau):
        return torch.exp(x / tau)


    def semi_loss_batch(self,h1,h2,temperature,batch_size,m1:Tensor):
        device = h1.device
        num_samples = h1.size(0)

        num_batches = (num_samples - 1) // batch_size + 1   
        num_batches = (num_samples - 1) // batch_size + 1
        indices = torch.arange(0, num_samples, device=device)   
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size: (i + 1) * batch_size]
            
            aug_positive = m1[mask]  # 100*500
            between_sim = self.f(self.cosine_similarity(h1[mask], h2), temperature)

            middle_sim = torch.mul(aug_positive, between_sim)[:, i * batch_size: (i + 1) * batch_size]
            # print(middle_sim)
            
            loss = -torch.log(0.5*(between_sim[:, i * batch_size: (i + 1) * batch_size].diag()+0.5*middle_sim.sum(1)) / between_sim.sum(1))
            # print(h1[mask].shape,h2.shape,between_sim.shape,loss.shape)
            # print("123",between_sim[:, i * batch_size: (i + 1) * batch_size].shape)
            # print(between_sim.sum(1).shape)
            losses.append(loss)
        # print("这个",torch.cat(losses).shape)
        return torch.cat(losses)
    

    def cosine_similarity(self, z1: Tensor, z2: Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())


if __name__ == "__main__":
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    knn_num=3
    path = "/root/0316/实验3/dataset/MSRC_v1.mat"
    data = loadmat(path)
    print(data.keys())

    X,y = data['fea'].T, data['gt']
    print(X.shape,y.squeeze(1)  )
    labels = (y-1).squeeze(1)    # size (2866,)
    print(labels)

    num_views = X.shape[0]
    num_samples = labels.size
    num_clusters = np.unique(labels).size

    print(f"输入数据集的视图个数为:{num_views}")
    print(f"输入数据集的样本个数为：{num_samples}")     # 输入数据集的样本个数为：4485
    print(f"数据集的聚类个数为：{num_clusters}")        # 数据集的聚类个数为：15


    hypergraph_views = list()
    data_views = list()
    input_sizes = list()           # 每个view的输入的维度


    for idx in range(num_views):
        if X[idx][0].shape[1] in (576,512):
            data_view = torch.from_numpy(X[idx][0].astype(np.float32)).to(device)
            input_sizes.append(X[idx][0].shape[1])
            print(data_view.shape)
            data_views.append(data_view)
            hypergraph_view = dhg.Hypergraph.from_feature_kNN(data_view, k=knn_num).to(device)
            hypergraph_views.append(hypergraph_view)


    # 只用PHOG和GIST两个视图
    num_views = 2 # 只用两个view哈
    # hypergraph_views = hypergraph_views 
    # data_views = data_views 
    # input_sizes = input_sizes[1:3]

    # 576 是HOG features
    # 512 是GIST features



    # 210个node

    h1 =torch.from_numpy(np.array( [[1,0],[1,1],[0,1]]))
    print(h1.shape)

    h1 =  hypergraph_view.H.to_dense()
    h2 = h1.T

    m1 = torch.mm(h1,h2)
    print(h1.shape,h2.shape,m1.shape)
    print(m1)

    m2 = torch.zeros_like(m1)
    print(m2.shape)
    m2[m1 > knn_num] = 1
    

    
    loss_func = ContrastiveLoss()
    h1 = torch.randn([210,512]).to(device=device)
    h2 = torch.randn([210,512]).to(device=device)
    temperature = 0.5
    batch_size = 200
    loss_func.batch_loss(h1,h2,temperature=temperature,batch_size=batch_size,m1=m1,m2=m2)


    # loss = MultiContrastiveLoss(temperature =0.5,batch_size=10)
    # z1,z2,h1,h2 = torch.randn([100,50]),torch.randn([100,50]),torch.randn([100,50]),torch.randn([100,50])
    # loss( z1,z2,h1,h2)
    '''
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func = ContrastiveLoss()
    h1 = torch.randn([500,512]).to(device=device)
    h2 = torch.randn([500,512]).to(device=device)
    temperature = 0.5
    batch_size = 100
    loss_func.batch_loss(h1,h2,temperature=temperature,batch_size=batch_size)
'''


'''
class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        # self.register_buffer("temperature", torch.tensor(temperature))
        # self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
        self.register_buffer("temperature", torch.tensor(temperature).to(device="cuda"))			# 超参数 温度
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device="cuda")).float())	
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
 
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

if __name__ == "__main__":
    loss_func = ContrastiveLoss(100)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_i = torch.randn([100,512]).to(device)
    emb_j = torch.randn([100,512]).to(device)

    print(loss_func(emb_i,emb_j))
# '''


# import torch
# from torch import nn
# import torch.nn.functional as F

# class ContrastiveLoss(nn.Module):
#     def __init__(self, batch_size, temperature=0.5):
#         super().__init__()
#         self.batch_size = batch_size
#         self.temperature = temperature
#         self.negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).float().to("cuda")

#     def forward(self, emb_i, emb_j):
#         z_i = F.normalize(emb_i, dim=1)
#         z_j = F.normalize(emb_j, dim=1)
#         representations = torch.cat([z_i, z_j], dim=0)

#         # 分批计算相似度矩阵
#         batch_size = self.batch_size
#         similarity_matrix = torch.zeros((2 * batch_size, 2 * batch_size), device=representations.device)
#         for i in range(0, 2 * batch_size, batch_size):
#             for j in range(0, 2 * batch_size, batch_size):
#                 similarity_matrix[i:i + batch_size, j:j + batch_size] = F.cosine_similarity(
#                     representations[i:i + batch_size].unsqueeze(1),
#                     representations[j:j + batch_size].unsqueeze(0),
#                     dim=2
#                 )

#         sim_ij = torch.diag(similarity_matrix, self.batch_size)
#         sim_ji = torch.diag(similarity_matrix, -self.batch_size)
#         positives = torch.cat([sim_ij, sim_ji], dim=0)

#         nominator = torch.exp(positives / self.temperature)
#         denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

#         loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
#         loss = torch.sum(loss_partial) / (2 * self.batch_size)
#         return loss

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     loss_func = ContrastiveLoss(100).to(device)
#     emb_i = torch.randn([100, 512]).to(device)
#     emb_j = torch.randn([100, 512]).to(device)

#     loss = loss_func(emb_i, emb_j)
#     print(loss)