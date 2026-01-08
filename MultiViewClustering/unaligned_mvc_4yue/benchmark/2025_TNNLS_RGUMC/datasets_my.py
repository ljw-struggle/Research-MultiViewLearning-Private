import os, random, sys
import numpy as np
import scipy.io as sio
import util
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler# 对数据标准化处理
tool = MinMaxScaler(feature_range=(0, 1))
import h5py


def load_data_my(config):
    """Load data """
    data_name = config['dataset']
    # main_dir = sys.path[0]
    X_list = []
    Y_list = []

    main_dir = os.path.dirname(os.getcwd()) # 上层目录

    if data_name in ['Scene_15']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', 'Scene-15.mat'))
        X = mat['X'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))

    elif data_name in ['Caltech101-20']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', data_name + '.mat'))
        X = mat['X'][0]
        for view in [3, 4]:
            x = X[view]
            x = util.normalize(x).astype('float32')
            X_list.append(x)
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)


    elif data_name in ['LandUse_21']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', 'LandUse-21.mat'))
        train_x = []
        train_x.append(sparse.csr_matrix(mat['X'][0, 0]).A)  # 20
        train_x.append(sparse.csr_matrix(mat['X'][0, 1]).A)  # 59
        train_x.append(sparse.csr_matrix(mat['X'][0, 2]).A)  # 40
        index = random.sample(range(train_x[0].shape[0]), 2100)
        for view in [1, 2]:
            x = train_x[view][index]
            X_list.append(x)
        y = np.squeeze(mat['Y']).astype('int')[index]
        Y_list.append(y)

    # elif data_name in ['NoisyMNIST']:
    #     data = sio.loadmat('./data/NoisyMNIST.mat')
    #     train = DataSet_NoisyMNIST(data['X1'], data['X2'], data['trainLabel'])
    #     tune = DataSet_NoisyMNIST(data['XV1'], data['XV2'], data['tuneLabel'])
    #     test = DataSet_NoisyMNIST(data['XTe1'], data['XTe2'], data['testLabel'])
    #     X_list.append(np.concatenate([tune.images1, test.images1], axis=0))
    #     X_list.append(np.concatenate([tune.images2, test.images2], axis=0))
    #     Y_list.append(np.concatenate([np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]))

    elif data_name in ['digit_2view']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', 'digit.mat'))
        X = mat['tmpX'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        X_list[0] = tool.fit_transform(X_list[0])
        X_list[1] = tool.fit_transform(X_list[1])
        Y_list.append(np.squeeze(mat['tmpY']))


    elif data_name in ['flower17_2views']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', data_name + '.mat'))
        X = mat['tmpX'][0]
        for view in [0, 1]:
            x = X[view]
            x = util.normalize(x).astype('float32')
            X_list.append(x)
        y = np.squeeze(mat['tmpY']).astype('int')
        Y_list.append(y)

    elif data_name in ['reuters_2views']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', data_name + '.mat'))
        X = mat['tmpX'][0]
        for view in range(2):
            x = X[view]
            x = util.normalize(x).astype('float32')
            X_list.append(x)
        y = np.squeeze(mat['tmpY']).astype('int')
        Y_list.append(y)

    elif data_name in ['AWA_2views']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', data_name + '.mat'))
        X = mat['tmpX'][0]
        for view in range(2):
            x = X[view]
            x = util.normalize(x).astype('float32')
            X_list.append(x)
        y = np.squeeze(mat['tmpY']).astype('int')
        Y_list.append(y)


    elif data_name in ['AWA_6views']:
        x1 = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data/AWA/tmpX1.mat'))
        x2 = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data/AWA/tmpX2.mat'))
        x3 = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data/AWA/tmpX3.mat'))
        x4 = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data/AWA/tmpX4.mat'))
        x5 = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data/AWA/tmpX5.mat'))
        x6 = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data/AWA/tmpX6.mat'))
        x1 = x1['tmpX1']
        x2 = x2['tmpX2']
        x3 = x3['tmpX3']
        x4 = x4['tmpX4']
        x5 = x5['tmpX5']
        x6 = x6['tmpX6']

        X= []
        X.append(x1)
        X.append(x2)
        X.append(x3)
        X.append(x4)
        X.append(x5)
        X.append(x6)

        for view in range(6):
            x = X[view]
            x = util.normalize(x).astype('float32')
            X_list.append(x)

        y = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data/AWA/AWA_label.mat'))
        Y_list.append(np.squeeze(y['AWA_label']))

    elif data_name in ['reuters_views_18758']:
        'test_completer_20220410/data'
        f = h5py.File('../test_completer_20220410/data/reuters_views_18758/data_EN_EN.mat')
        f_t = np.array(f['data_EN_EN'])
        X_list.append(f_t.astype('float32'))
        f = h5py.File('../test_completer_20220410/data/reuters_views_18758/data_EN_FR.mat')
        f_t = np.array(f['data_EN_FR'])
        X_list.append(f_t.astype('float32'))
        for v in range(2):
            X_list[v] = np.transpose(X_list[v])
            X_list[v] = tool.fit_transform(X_list[v])
        # X_list[1] = tool.fit_transform(X_list[1])
        y = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', data_name + '/label_EN.mat'))
        Y_list.append(np.squeeze(y['label_EN']))




    elif data_name in ['YouTuBe']:
        mat1 = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', data_name + '/YouTuBe-cut31/train_data_cut31.mat'))
        mat2 = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', data_name + '/YouTuBe-cut31/test_data_cut31.mat'))
        mat3 = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', data_name + '/YouTuBe-cut31/val_data_cut31.mat'))
        X1 = mat1['data_train'][0]
        X2 = mat2['data_test'][0]
        X3 = mat3['data_val'][0]
        view1 = np.vstack((X1[0], X2[0], X3[0]))  # 只需要训练、测试、验证集中的2个视图
        view2 = np.vstack((X1[1], X2[1], X3[1]))
        X_list.append(view1.astype('float32'))
        X_list.append(view2.astype('float32'))
        label = np.vstack((mat1['label_train'], mat2['label_test'], mat3['label_val']))
        Y_list.append(np.squeeze(label))
        print('lable', len(Y_list[0]))


    elif data_name in ['digit_5view']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', 'digit_5views.mat'))
        X = mat['tmpX'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        X_list.append(X[2].astype('float32'))
        X_list.append(X[3].astype('float32'))
        X_list.append(X[4].astype('float32'))

        X_list[0] = tool.fit_transform(X_list[0])
        X_list[1] = tool.fit_transform(X_list[1])
        X_list[2] = tool.fit_transform(X_list[2])
        X_list[3] = tool.fit_transform(X_list[3])
        X_list[4] = tool.fit_transform(X_list[4])

        Y_list.append(np.squeeze(mat['tmpY']))

    elif data_name in ['digit_6view']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', 'digit_6views.mat'))
        X = mat['tmpX'][0]
        for v in range(6):
            X_list.append(X[v].astype('float32'))
            X_list[v] = tool.fit_transform(X_list[v])
        Y_list.append(np.squeeze(mat['tmpY']))

    elif data_name in ['Caltech101-20-5views']:

        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', 'Caltech101-20-5views.mat'))
        X = mat['tmpX'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        X_list.append(X[2].astype('float32'))
        X_list.append(X[3].astype('float32'))
        X_list.append(X[4].astype('float32'))

        X_list[0] = tool.fit_transform(X_list[0])
        X_list[1] = tool.fit_transform(X_list[1])
        X_list[2] = tool.fit_transform(X_list[2])
        X_list[3] = tool.fit_transform(X_list[3])
        X_list[4] = tool.fit_transform(X_list[4])

        Y_list.append(np.squeeze(mat['tmpY']))

    elif data_name in ['flower17_5views']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', 'flower17_5views.mat'))
        X = mat['tmpX'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        X_list.append(X[2].astype('float32'))
        X_list.append(X[3].astype('float32'))
        X_list.append(X[4].astype('float32'))

        X_list[0] = tool.fit_transform(X_list[0])
        X_list[1] = tool.fit_transform(X_list[1])
        X_list[2] = tool.fit_transform(X_list[2])
        X_list[3] = tool.fit_transform(X_list[3])
        X_list[4] = tool.fit_transform(X_list[4])

        Y_list.append(np.squeeze(mat['tmpY']))

    elif data_name in ['reuters']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', 'reuters.mat'))
        X = mat['tmpX'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        X_list.append(X[2].astype('float32'))
        X_list.append(X[3].astype('float32'))
        X_list.append(X[4].astype('float32'))

        X_list[0] = tool.fit_transform(X_list[0])
        X_list[1] = tool.fit_transform(X_list[1])
        X_list[2] = tool.fit_transform(X_list[2])
        X_list[3] = tool.fit_transform(X_list[3])
        X_list[4] = tool.fit_transform(X_list[4])

        Y_list.append(np.squeeze(mat['tmpY']))


    elif data_name in ['flower17_7views']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', 'flower17_7views.mat'))
        X = mat['tmpX'][0]
        for v in range(7):
            X_list.append(X[v].astype('float32'))
            X_list[v] = tool.fit_transform(X_list[v])

        Y_list.append(np.squeeze(mat['tmpY']))

    elif data_name in ['reuters_views_18758_5views']:
        'test_completer_20220410/data'
        f = h5py.File('../test_completer_20220410/data/reuters_views_18758/data_EN_EN.mat')
        f_t = np.array(f['data_EN_EN'])
        X_list.append(f_t.astype('float32'))
        f = h5py.File('../test_completer_20220410/data/reuters_views_18758/data_EN_FR.mat')
        f_t = np.array(f['data_EN_FR'])
        X_list.append(f_t.astype('float32'))
        f = h5py.File('../test_completer_20220410/data/reuters_views_18758/data_EN_GR.mat')
        f_t = np.array(f['data_EN_GR'])
        X_list.append(f_t.astype('float32'))
        f = h5py.File('../test_completer_20220410/data/reuters_views_18758/data_EN_IT.mat')
        f_t = np.array(f['data_EN_IT'])
        X_list.append(f_t.astype('float32'))
        f = h5py.File('../test_completer_20220410/data/reuters_views_18758/data_EN_SP.mat')
        f_t = np.array(f['data_EN_SP'])
        X_list.append(f_t.astype('float32'))

        for v in range(5):
            X_list[v] = np.transpose(X_list[v])
            X_list[v] = tool.fit_transform(X_list[v])
        # X_list[1] = tool.fit_transform(X_list[1])
        # y = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', data_name + '/label_EN.mat'))#
        y = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data/reuters_views_18758/label_EN.mat'))
        Y_list.append(np.squeeze(y['label_EN']))


    # elif data_name in ['reuters_views_18758_5views']:
    #     'test_completer_20220410/data'
    #     f = h5py.File('../test_completer_20220410/data/reuters_views_18758/data_EN_EN.mat')#21531
    #     f_t = np.array(f['data_EN_EN'])
    #     X_list.append(f_t.astype('float32'))
    #     f = h5py.File('../test_completer_20220410/data/reuters_views_18758/data_EN_FR.mat')#24892
    #     f_t = np.array(f['data_EN_FR'])
    #     X_list.append(f_t.astype('float32'))
    #     f = h5py.File('../test_completer_20220410/data/reuters_views_18758/data_EN_GR.mat')#34251
    #     f_t = np.array(f['data_EN_GR'])
    #     X_list.append(f_t.astype('float32'))
    #     f = h5py.File('../test_completer_20220410/data/reuters_views_18758/data_EN_IT.mat')#15506
    #     f_t = np.array(f['data_EN_IT'])
    #     X_list.append(f_t.astype('float32'))
    #     # f = h5py.File('../test_completer_20220410/data/reuters_views_18758/data_EN_SP.mat')#11547
    #     # f_t = np.array(f['data_EN_SP'])
    #     # X_list.append(f_t.astype('float32'))
    #     for v in range(5):
    #         X_list[v] = np.transpose(X_list[v])
    #         X_list[v] = tool.fit_transform(X_list[v])
    #     # X_list[1] = tool.fit_transform(X_list[1])
    #     y = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', data_name + '/label_EN.mat'))
    #     Y_list.append(np.squeeze(y['label_EN']))

    elif data_name in ['Caltech101-20-6view']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', 'Caltech101-20.mat'))
        X = mat['X'][0]
        for view in range(6):
            x = X[view]
            x = util.normalize(x).astype('float32')
            X_list.append(x)
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)

    # elif data_name in ['Scene_15']:
    #         mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', 'Scene-15.mat'))

    #
    elif data_name in ['Scene_15-3views']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', 'Scene-15.mat'))
        X = mat['X'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        X_list.append(X[2].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))

    elif data_name in ['office31_resnet50']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', data_name + '/tmpX.mat'))
        X = mat['tmpX'][0]# mat['tmpX'][0]
        for view in range(3):
            x = X[view]
            x = util.normalize(x).astype('float32')
            X_list.append(x)

        y = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', data_name + '/tmpY.mat'))
        Y_list.append(np.squeeze(y))

    elif data_name in ['YaleB']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data/YaleB/YaleB_first10.mat'))
        X_list.append(mat['X1'].astype('float32').T)
        X_list.append(mat['X2'].astype('float32').T)
        X_list.append(mat['X3'].astype('float32').T)

        y = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data/YaleB/lable.mat'))
        Y_list.append(np.squeeze(y['lable']).astype('int'))

    elif data_name in ['YaleB_2views']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data/YaleB/YaleB_first10.mat'))
        X_list.append(mat['X1'].astype('float32').T)
        X_list.append(mat['X2'].astype('float32').T)

        y = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data/YaleB/lable.mat'))
        Y_list.append(np.squeeze(y['lable']).astype('int'))

    elif data_name in ['reuters_views_111740_5view']:

        f = h5py.File('./data/rcv1rcv2aminigoutte/unpaired_dataset2/data_EN_EN.mat')
        f_t = np.array(f['data_EN_EN'])
        X_list.append(f_t.astype('float32'))
        f = h5py.File('./data/rcv1rcv2aminigoutte/unpaired_dataset2/data_FR_FR.mat')
        f_t = np.array(f['data_FR_FR'])
        X_list.append(f_t.astype('float32'))
        f = h5py.File('./data/rcv1rcv2aminigoutte/unpaired_dataset2/data_GR_GR.mat')
        f_t = np.array(f['data_GR_GR'])
        X_list.append(f_t.astype('float32'))
        f = h5py.File('./data/rcv1rcv2aminigoutte/unpaired_dataset2/data_IT_IT.mat')
        f_t = np.array(f['data_IT_IT'])
        X_list.append(f_t.astype('float32'))
        f = h5py.File('./data/rcv1rcv2aminigoutte/unpaired_dataset2/data_SP_SP.mat')
        f_t = np.array(f['data_SP_SP'])
        X_list.append(f_t.astype('float32'))

        y = sio.loadmat(os.path.join(main_dir, 'data', 'rcv1rcv2aminigoutte/unpaired_dataset/label_allsample.mat'))
        Y_list.append(np.squeeze(y['label_allsample']))

        for v in range(5):
            X_list[v] = np.transpose(X_list[v])
            X_list[v] = tool.fit_transform(X_list[v])
        # f = h5py.File('./data/rcv1rcv2aminigoutte/unpaired_dataset/data1.mat')
        # print(f.keys())
        # print(f.values())
        # f_t = np.array(f['data'])
        # print(f_t)
        # print(f['data'].shape)
        # X_list.append(f['data'][0][0])
        # X_list.append(f[0][0].astype('float32'))
        print(X_list[0].shape)

    return X_list, Y_list

