import dhg
import torch
def build_graph(latent_data_views,knn_num):
    hypergraph_views = list()
    num_views = len(latent_data_views)
    for idx in range(num_views):
        latent_data_view = latent_data_views[idx]
        device = latent_data_view.device
        hypergraph_view = dhg.Hypergraph.from_feature_kNN(latent_data_view, k=knn_num).to(device)
        hypergraph_views.append(hypergraph_view)
        print(hypergraph_view.L_HGNN)
    return hypergraph_views




if __name__=="__main__":
    latent_data_views = list()
    Z = torch.randn(10,5)
    latent_data_views.append(Z)
    hypergraph_views = build_graph(latent_data_views,3)

    L_HGNN = hypergraph_views[0].L_HGNN
    L_dense = L_HGNN.to_dense()
    reg_dense = torch.trace(Z.T @ L_dense @ Z)
    print(reg_dense)


#     import torch

# # 假设 L_HGNN 是你的稀疏矩阵（形状 10x10）
# L_HGNN = torch.sparse_coo_tensor(
#     indices=torch.tensor([[0, 0, 0, ...], [0, 2, 3, ...]]),
#     values=torch.tensor([0.3333, 0.0962, ...]),
#     size=(10, 10)
# )

# # 转换为稠密矩阵（可选，稀疏乘法也可直接支持）
# L_dense = L_HGNN.to_dense()

# # 计算 Z L Z^T，再取迹
# regularizer = torch.trace(Z @ L_dense @ Z.T)  # 或使用稀疏乘法优化