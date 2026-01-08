from scipy.io import loadmat
import os
import argparse

import numpy as np
import time
import torch
from network import Autoencoder
from HGNN import Encoder
from losses import MultiContrastiveLoss
from build_graph import build_graph
from sklearn.cluster import KMeans
from evaluation import eva
from dataloader import load_data


import logging

Dataname = 'BBCSport'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument("--latent_dim", default=256)
parser.add_argument("--pretrain_epoch_num", default=3000)
parser.add_argument("--finetune_epoch_num", default=300)
parser.add_argument("--dim_high_feature", default=512)
parser.add_argument("--dim_proj_feature", default=512)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--temperature", default=0.1)



args = parser.parse_args()

log_dir = os.path.abspath('../logs')
os.makedirs(log_dir, exist_ok=True)
# 配置日志文件
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别（INFO/WARNING/ERROR/DEBUG）
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
    filename=os.path.join(log_dir, f'{Dataname}.log'),  # 日志文件名
    filemode='a'  # 'a' 表示追加模式，'w' 表示覆盖模式
)

# 获取 logger 对象
logger = logging.getLogger(__name__)
torch.manual_seed(0) # 设置随机种子


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
step 1 数据读取
'''
data_views,labels,input_sizes,num_views,num_samples,num_clusters = load_data(args.dataset)
print(f"输入数据集的视图个数为:{num_views}")
print(f"输入数据集的样本个数为：{num_samples}")      
print(f"数据集的聚类个数为：{num_clusters}")         
print(f"输入的每个视图的数据维度为：{input_sizes}")

'''
step 2 数据预训练得到latent embedding
'''

if args.dataset == "BBCSport":
    args.latent_dim=256
    args.temperature = 0.1
    args.pretrain_lr = 0.001
elif args.dataset == "WebKB":
    args.temperature = 0.1
    args.pretrain_lr = 0.001


latent_data_views = list()
st = time.time()
for idx in range(num_views):
    data_view = data_views[idx]
    input_dim = input_sizes[idx]
    output_dim = args.latent_dim
    model = Autoencoder(input_dim, output_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    criterion = torch.nn.MSELoss() 
    loss_list = []
    lr_list = []
    
    for epoch in range(args.pretrain_epoch_num):
        optimizer.zero_grad()
        x_hat, z = model(data_view)
        loss = criterion(x_hat, data_view)
        loss.backward()
        optimizer.step()
        
        scheduler.step()
        lr_list.append(optimizer.param_groups[0]['lr'])

        loss_list.append(loss.item())
        print(f'Epoch {epoch+1}/{args.pretrain_epoch_num}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.8f}')

    model.eval()
    with torch.no_grad():
        x_hat, z = model(data_view)
        latent_data_views.append(z)
print("预训练时间", time.time()-st)



'''
3、微调多视图聚类部分
'''
# grid search参数范围，可调
param2_list = [0,0.001,0.01,1,5]
knn_num_list=[20,15,10,5,3]
param1_list =  [5,4,3,2,1]
param3_list = [0.0001,0.001,0,1,5,0.0002,0.0003,0.0004,0.0005]


def train(param1,param2,param3,knn_num):
    initial_hypergraph_views = build_graph(data_views,knn_num=knn_num)   # 原始特征构造的超图视图
    hypergraph_views = build_graph(latent_data_views,knn_num=knn_num)
    logging.info(f"param1: {param1}, param2: {param2}, param3: {param3},knn_num:{knn_num}")
    hg_model = Encoder(num_views,input_dim = args.latent_dim,dim_high_feature=args.dim_high_feature,dim_proj_feature=args.dim_proj_feature)
    hg_model = hg_model.to(device)
    hg_model.train()

    optimizer = torch.optim.Adam(hg_model.parameters(), lr=args.learning_rate, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    mvc_loss = MultiContrastiveLoss(temperature=0.5,batch_size=num_samples)
    loss_list = []
    st = time.time()

    initial_L_HGNN_list = []
    for i in range(num_views):
        initial_L_HGNN_list.append(initial_hypergraph_views[i].L_HGNN.to_dense())

    for epoch in range(args.finetune_epoch_num):
        optimizer.zero_grad()
        node_encoder_output_features,node_proj_output_features = hg_model(latent_data_views,hypergraph_views)
        # loss_inter_view
        loss_epoch = []
        for i in range(num_views):
            for j in range(num_views):
                if i != j:
                    loss_epoch.append(mvc_loss.compute_contrastive_loss(node_proj_output_features[i],node_proj_output_features[j],hypergraph_views[i],hypergraph_views[j],tau=args.temperature,batch_size=num_samples)   )
        loss_inter = torch.stack(loss_epoch).mean()
        
        # loss_intra_view
        loss_intra = []  # 计算每个视图的intra-view loss
        for i in range(num_views):
            loss_intra.append(mvc_loss.compute_inner_view_loss2(node_proj_output_features[i],hypergraph_views[i],tau=args.temperature,batch_size=num_samples))
        loss_intra = torch.stack(loss_intra).mean()

        # loss_regularizer
        reg_dense_list = [] # 计算每个视图的regularizer loss
        for i in range(num_views):
            reg_dense_list.append(torch.trace(node_encoder_output_features[i].T@ initial_L_HGNN_list[i] @node_encoder_output_features[i]) )
        loss_reg = torch.stack(reg_dense_list).mean()

        loss = param1*loss_intra + param2*loss_inter + param3*loss_reg

        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_list.append(loss.item())
        print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")

    # 新增聚类层
    hg_model.eval()
    with torch.no_grad():
        node_encoder_output_features,node_proj_output_features = hg_model(latent_data_views,hypergraph_views)
    
    # fused_x = torch.cat([node_encoder_output_features[0],node_encoder_output_features[1]], dim=1)    

    fused_x = torch.cat([node_encoder_output_features[i] for i in range(num_views)],dim=1)

    fused_x = fused_x.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=num_clusters)#init Kmeans 
    y_pred = kmeans.fit_predict(fused_x)# fit 
    cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(device)

    # plot_tsne(fused_x , y_pred,"pre", "/root/0316/tmp/1预训练分布-.jpg")

    acc, nmi, ari, f1 = eva(labels, y_pred, show_details=False)
    print("1\预训练以后的kmeans聚类结果：result:{:.4f}, {:.4f}, {:.4f}, {:.4f}".format(acc, nmi, ari, f1))

    logging.info("聚类结果：result:{:.4f}, {:.4f}, {:.4f}, {:.4f}".format(acc, nmi, ari, f1))
    logging.info("**********************end training*******************")
from itertools import product
for param1, param2, param3,knn_num in product(param1_list, param2_list, param3_list,knn_num_list):
    if param1 + param2 +param3 >0:
        train(param1,param2,param3,knn_num)
