'''
This method and code is activate by works:completer+SiMVC:  Loss: Autoencoder_loss
silhouette coefficient: choose the anchor sample;
'''

import argparse
import collections
import itertools
from model_baseline_v4_20230530 import *  # 单个视图指引  +  多个视图指引
from get_mask_my import get_mask_my #,get_mask_shuffle_my
from util import cal_std, get_logger, cal_std_my
from datasets_my import *
from configure_my import get_default_config#
from sklearn.preprocessing import StandardScaler  #样本数据归一化，标准化



# dataset
dataset = {
    0: "Caltech101-20",
    1: "Scene_15",
    2: "LandUse_21",
    3: "NoisyMNIST",
    4: "digit_2view",
    5: "flower17_2views",
    6: "reuters_2views",
    7: "AWA_2views",
    8: "YouTuBe",
    9: "office31_2views",
    10: "reuters_views_18758",
}
# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='4', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='10', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='1', help='number of test times')
parser.add_argument('--tau', type=float, default='0.1', help='hyperparameter in contrastive learning')
args = parser.parse_args()
dataset = dataset[args.dataset]

def main():
    use_cuda = torch.cuda.is_available() #检验当前是否有GPU可用
    device = torch.device('cuda:0' if use_cuda else 'cpu')# 默认选择 第2张卡

    # Configure
    config = get_default_config(dataset)# Scene_15
    config['print_num'] = args.print_num# 100
    config['dataset'] = dataset
    config['test_time'] = args.test_time
    logger = get_logger()


    # Load data
    # complete sample
    X_list, Y_list = load_data_my(config)
    x_train_raw1 = X_list.copy()
    view_num = len(X_list)
    Y_label = Y_list.copy()


    accumulated_metrics = collections.defaultdict(list)# 为字典提供默认值，
    best_result_metrics = collections.defaultdict(list)  # 为字典提供默认值，
    paired_rate = 0

    for data_seed in range(1, args.test_time + 1):
        # Get the Mask; 得到缺失样本对应的下标
        np.random.seed(data_seed)
        mask, tmp_idx, mis_idx = get_mask_my(view_num, Y_label, paired_rate)
        # 对于数据集本身类别对应样本数目不一样高，从导致构造不同视图构造样本数目不一致的问题，做一些微调
        if len(tmp_idx[0]) > len(tmp_idx[1]):
            tmp_idx[1] = np.hstack((tmp_idx[1], tmp_idx[0][-int(np.abs(len(tmp_idx[0])-len(tmp_idx[1]))/2):]))
            tmp_idx[0] = tmp_idx[0][0: len(tmp_idx[0])-int(np.abs(len(tmp_idx[0])-len(tmp_idx[1])))]
        elif len(tmp_idx[0]) < len(tmp_idx[1]):
            tmp_idx[0] = np.hstack((tmp_idx[0], tmp_idx[1][-int(np.abs(len(tmp_idx[0]) - len(tmp_idx[1])) / 2):]))
            tmp_idx[1] = tmp_idx[1][0: len(tmp_idx[1])-int(np.abs(len(tmp_idx[0])-len(tmp_idx[1])))]


        x_train_raw = []
        Y_view = []
        for v in range(0, view_num):
            a = np.array(tmp_idx[v]).astype(int)
            x_train_raw.append(x_train_raw1[v][a])
            Y_view.append(Y_label[0][a])
        Y_label_adjust = np.concatenate([Y_view[0], Y_view[1]], axis=0)



        mask = torch.from_numpy(mask).long().to(device)

        # Set random seeds
        if config['training']['missing_rate'] == 0:
            seed = data_seed
        else:
            seed = config['training']['seed']
        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.backends.cudnn.deterministic = True

        COMPLETER_my = Completer_my_20220605(config)

        optimizer = torch.optim.Adam(
            itertools.chain(COMPLETER_my.autoencoder1.parameters(), COMPLETER_my.autoencoder2.parameters(),
                            COMPLETER_my.img2txt.parameters(), COMPLETER_my.txt2img.parameters()),
            lr=config['training']['lr'])
        COMPLETER_my.to_device(device)

        # Print the models
        logger.info(COMPLETER_my.autoencoder1)
        logger.info(COMPLETER_my.img2txt)
        logger.info(optimizer)

        print('oktoday')

        standardscaler = StandardScaler()
        # 对数组x遍历，对每一个样本进行标准化
        # 只对观测样本标准化
        x_train = []
        for v in range(view_num):
            scaler = standardscaler.fit(x_train_raw[v])  # 存储计算得到的均值和方差
            x_train.append(scaler.transform(x_train_raw[v]))  # 返回标准化后的样本集

        acc, nmi, pre, f_measure, recall, ari, ami, acc_show, nmi_show, precision_show, F_measure_show, recall_show, ARI_show, AMI_show, \
        result_single_view1_all, result_single_view2_all \
            = COMPLETER_my.train(config, logger, x_train, Y_label_adjust, mask, optimizer, device, tmp_idx)
        print('result_single_view1_all, result_single_view2_all ', result_single_view1_all, result_single_view2_all) # 20230411


        accumulated_metrics['acc'].append(acc)
        accumulated_metrics['nmi'].append(nmi)
        accumulated_metrics['Precision'].append(pre)
        accumulated_metrics['F_score'].append(f_measure)
        accumulated_metrics['Recall'].append(recall)
        accumulated_metrics['ari'].append(ari)
        accumulated_metrics['AMI'].append(ami)
        sio.savemat('accumulated.mat', {'accumulated_metrics': accumulated_metrics})

    logger.info('--------------------Training over--------------------')
    cal_std_my(logger, accumulated_metrics['acc'], accumulated_metrics['nmi'],
               accumulated_metrics['Precision'],
               accumulated_metrics['F_score'], accumulated_metrics['Recall'],
               accumulated_metrics['ari'], accumulated_metrics['AMI'],)
    accumulated_metrics_copy = accumulated_metrics
    sio.savemat('accumulated_copy.mat', {'accumulated_metrics_copy': accumulated_metrics_copy})
    t_show = np.sum([acc_show, nmi_show, F_measure_show], axis=0)
    t = np.where(t_show == np.max(t_show))
    best_result_idx = list(t)[0][0]
    print('best_result_idx: ', best_result_idx)

    best_result = [acc_show[best_result_idx], nmi_show[best_result_idx], precision_show[best_result_idx],
                   F_measure_show[best_result_idx],
                   recall_show[best_result_idx], ARI_show[best_result_idx], AMI_show[best_result_idx]]
    print('best_result(acc, nmi, pre, F-measure, recall, ARI, AMI): ', np.array(best_result)*100)

if __name__ == '__main__':
    main()