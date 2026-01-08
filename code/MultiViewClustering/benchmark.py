from _benchmark import benchmark_2025_TNNLS_MCMC
from _benchmark import benchmark_2025_NIPS_SparseMVC
# from _benchmark import benchmark_2025_ICCV_RML
from _benchmark import benchmark_2024_TMM_SCMVC
# from _benchmark import benchmark_2024_INFU_MAGA
# from _benchmark import benchmark_2024_IJCAI_SCM
from _benchmark import benchmark_2024_CVPR_MVCAN
# from _benchmark import benchmark_2023_TKDE_SDMVC --
# from _benchmark import benchmark_2023_NIPS_SEM
from _benchmark import benchmark_2023_MM_DealMVC
from _benchmark import benchmark_2023_ICCV_CVCL
from _benchmark import benchmark_2023_CVPR_GCFAgg
from _benchmark import benchmark_2022_CVPR_MFLVC
# from _benchmark import benchmark_2022_CVPR_DSMVC
# from _benchmark import benchmark_2021_INSC_DEMVC --
# from _benchmark import benchmark_2021_ICCV_MultiVAE --
# from _benchmark import benchmark_2021_CVPR_SiMVC_CoMVC --
from _benchmark import benchmark_2017_IJCAI_IDEC
from _benchmark import benchmark_2016_PMLR_DEC

if __name__ == "__main__":
    acc, nmi, pur, ari = benchmark_2025_TNNLS_MCMC(dataset_name='BDGP', batch_size=128, temperature_f=0.5, temperature_l=1.0, learning_rate=0.0005, 
                                                   weight_decay=0., mse_epochs=200, con_epochs=50, feature_dim=256, high_feature_dim=128, seed=10)
    acc, nmi, pur, ari = benchmark_2025_NIPS_SparseMVC(dataset_name='MSRCV1', batch_size=256, learning_rate=0.0003, pre_epochs=300, con_epochs=300, 
                                                       feature_dim=64, high_feature_dim=20, seed=50, weight_decay=0.0)
    acc, nmi, pur, ari = benchmark_2024_TMM_SCMVC(dataset_name='MNIST-USPS', batch_size=256, learning_rate=0.0003, weight_decay=0., pre_epochs=200, con_epochs=50, 
                                                  feature_dim=64, high_feature_dim=20, temperature=1, seed=10)
    acc, nmi, pur, ari = benchmark_2024_CVPR_MVCAN(dataset_name='BDGP', batch_size=100, pre_epochs=100, train_epochs=100, T_1=0.5, T_2=1.0, lr=0.0003, lambda_clu=1.0, seed=10)
    acc, nmi, pur, ari = benchmark_2023_ICCV_CVCL(dataset_name='MSRCv1', learning_rate=0.0005, weight_decay=0., batch_size=100, seed=10, mse_epochs=200, con_epochs=100, 
                                                   normalized=False, temperature_l=1.0, dim_high_feature=128, dim_low_feature=64, hidden_dims=[64, 64], alpha=0.5, beta=0.5)
    acc, nmi, pur, ari = benchmark_2022_CVPR_MFLVC(dataset_name='MNIST-USPS', batch_size=256, temperature_f=0.5, temperature_l=1.0, learning_rate=0.0003, 
                                                   weight_decay=0., mse_epochs=200, con_epochs=50, tune_epochs=50, feature_dim=512, high_feature_dim=128, seed=10)
    acc, nmi, pur, ari = benchmark_2023_MM_DealMVC(dataset_name='Hdigit', batch_size=256, temperature_f=0.5, temperature_l=1.0, learning_rate=0.0003, 
                                                   weight_decay=0., mse_epochs=200, con_epochs=50, tune_epochs=50, feature_dim=512, high_feature_dim=128, seed=10, 
                                                   threshold=0.8, num_heads=8, hidden_dim=256, dropout_rate=0.5, ffn_hidden_dim=32)
    acc, nmi, pur, ari = benchmark_2023_CVPR_GCFAgg(dataset_name='Hdigit', batch_size=256, temperature_f=0.5, learning_rate=0.0003, weight_decay=0., 
                                                    mse_epochs=200, tune_epochs=100, low_feature_dim=512, high_feature_dim=128, seed=10)
