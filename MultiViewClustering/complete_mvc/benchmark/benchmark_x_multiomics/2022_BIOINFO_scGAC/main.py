import os, time, argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from keras import backend as K
from keras import activations, constraints, initializers, regularizers
from keras.layers import Input, Dropout, Layer, LeakyReLU, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

class GraphAttention(Layer):
    # This implementation follows https://github.com/danielegrattarola/keras-gat/tree/master/keras_gat, which is released under MIT License.
    def __init__(self, hidden_dim, attn_heads=1, attn_heads_reduction='concat', dropout_rate=0.5, activation='relu',
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', attn_kernel_initializer='glorot_uniform', 
                 kernel_regularizer=None, bias_regularizer=None, attn_kernel_regularizer=None, activity_regularizer=None, 
                 kernel_constraint=None, bias_constraint=None, attn_kernel_constraint=None, **kwargs):
        assert attn_heads_reduction in {'concat', 'average'}, 'Possbile reduction methods: concat, average'
        self.hidden_dim = hidden_dim; self.attn_heads = attn_heads; self.attn_heads_reduction = attn_heads_reduction; self.dropout_rate = dropout_rate; self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer); self.bias_initializer = initializers.get(bias_initializer); self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer); self.bias_regularizer = regularizers.get(bias_regularizer); self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer); self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint); self.bias_constraint = constraints.get(bias_constraint); self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.kernels = []; self.biases = []; self.attn_kernels = []  # Initialize weights for each attention head include kernel, bias, and attention kernel
        self.dropout_att = Dropout(self.dropout_rate); self.dropout_feat = Dropout(self.dropout_rate)
        self.output_dim = self.hidden_dim * self.attn_heads if attn_heads_reduction == 'concat' else self.hidden_dim
        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2 # input_shape[0] is the node features, input_shape[1] is the adjacency matrix
        input_dim = input_shape[0][-1]
        for head in range(self.attn_heads): # Initialize weights for each attention head
            kernel = self.add_weight(shape=(input_dim, self.hidden_dim), initializer=self.kernel_initializer, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint, name='kernel_{}'.format(head)); self.kernels.append(kernel)
            bias = self.add_weight(shape=(self.hidden_dim, ), initializer=self.bias_initializer, regularizer=self.bias_regularizer, constraint=self.bias_constraint, name='bias_{}'.format(head)); self.biases.append(bias)
            attn_kernel_self = self.add_weight(shape=(self.hidden_dim, 1), initializer=self.attn_kernel_initializer, regularizer=self.attn_kernel_regularizer, constraint=self.attn_kernel_constraint, name='attn_kernel_self_{}'.format(head),)
            attn_kernel_neighs = self.add_weight(shape=(self.hidden_dim, 1), initializer=self.attn_kernel_initializer, regularizer=self.attn_kernel_regularizer, constraint=self.attn_kernel_constraint, name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs, training=None):
        X = inputs[0]; A = inputs[1]  # Node features (N x input_dim), Adjacency matrix (N x N)
        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]; attention_kernel = self.attn_kernels[head]; bias = self.biases[head]
            features = K.dot(X, kernel)  # (N x hidden_dim), Apply linear transformation to node features
            attn_for_self = K.dot(features, attention_kernel[0])  # (N x 1), Apply attention kernel to node features 
            attn_for_neighs = K.dot(features, attention_kernel[1])  # (N x 1), Apply attention kernel to node features
            dense = K.exp(-K.square(attn_for_self - K.transpose(attn_for_neighs))/1e-0) # (N x N), Calculate attention coefficients (mechanism of attention: the more similar the features of two nodes are, the more attention they will pay to each other)
            dense = K.exp(dense) * A  # (N x N), Calculate attention coefficients with mask (mask is the adjacency matrix)
            attn = K.transpose(K.transpose(dense) / K.sum(dense, axis=1)) # (N x N), Normalize attention coefficients
            K.sum(attn, axis=1, keep_dims=True) # (N x 1), Sum of attention coefficients
            attn = self.dropout_att(attn, training=training)  # (N x N), Apply dropout to attention coefficients
            features = self.dropout_feat(features, training=training)  # (N x hidden_dim), Apply dropout to features
            node_features = K.dot(attn, features)  # (N x hidden_dim), Linear combination with neighbors' features
            node_features = K.bias_add(node_features, bias) # (N x hidden_dim), Add bias
            outputs.append(node_features)
        output = K.concatenate(outputs) if self.attn_heads_reduction == 'concat' else K.mean(K.stack(outputs), axis=0) # shape: (N x hidden_dim * attn_heads) if 'concat' or (N x hidden_dim) if 'average'
        output = self.activation(output)
        # self.add_loss(self.activity_regularizer(output)) if self.activity_regularizer else None # Add activity regularizer
        return output # (N x hidden_dim * attn_heads) if 'concat' or (N x hidden_dim) if 'average'

    def compute_output_shape(self, input_shape): # this fuction is used when calling model.summary()
        output_shape = input_shape[0][0], self.output_dim
        return output_shape

class ClusteringLayer(Layer):
    def __init__(self, n_clusters, clusters=None, mode='q', threshold = 0.1, alpha=1.0, **kwargs):
        self.n_clusters = n_clusters; self.clusters = clusters; self.mode = mode; self.threshold = threshold; self.alpha = alpha # clusters.shape: (n_clusters, n_features)
        super(ClusteringLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs): # inputs: hidden representation of cells, shape: (N x hidden_dim)
        # 1\ Calculate the probability of cell i belonging to cluster k: 
        # q_ik = (1 + ||z_i - miu_k||^2)^-1, where z_i is the hidden representation of cell i, miu_k is the cluster center of cluster k. 
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha)) # (N x n_clusters)
        # 2\ Calculate the normalized probability of cell i belonging to cluster k:
        # q_ik = q_ik / sum(q_ik, dim=1), where sum(q_ik, dim=1) is the sum of q_ik over all clusters.
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # (N x n_clusters)
        # 3\ Update the cluster center:
        # if self.mode == 'q2':
        #     q_ = q ** 2 / K.sum(q, axis=0) # (N x K), q_ik = q_ik^2 / sum(q_ik, dim=0), where sum(q_ik, dim=0) is the sum of q_ik over all cells.
        #     q_ = K.transpose(K.transpose(q_) / K.sum(q_, axis=1)) # (N x n_clusters), q_ik = q_ik / sum(q_ik, dim=1), where sum(q_ik, dim=1) is the sum of q_ik over all clusters.
        # else:
        #     q_ = q + 1e-20 # (N x n_clusters)
        q_ = q + 1e-20 # (N x n_clusters)
        q_ = K.one_hot(K.argmax(q_, axis=1), self.n_clusters) * q_ # shape: (N, n_clusters), q_ik = q_ik if k = argmax(q_ik) else 0
        q_ = K.relu(q_ - self.threshold) # (N x n_clusters), q_ik = q_ik - threshold if q_ik > threshold else 0
        q_ = q_ + K.sign(q_) * self.threshold # (N x n_clusters), q_ik = 0 if q_ik = 0 else q_ik + threshold
        self.clusters = K.dot(K.transpose(q_ / K.sum(q_, axis=0)), inputs) # (n_clusters x hidden_dim), Update cluster center by weighted sum of hidden representation of cells
        return q # (N x n_clusters)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

############################################################################################################################################################################

def NE_dn(w, type='ave'): # Denoise the weight matrix
    # w: weight matrix (symmetric matrix, w.shape=(num_cells, num_cells)); type: denoise type, 'ave' or 'gph'
    D = np.sum(np.abs(w), axis=1) + 1e-20 # D: degree matrix, D.shape=(num_cells,)
    if type == 'ave': D_ = np.diag(1 / D); w_normal = np.dot(D_, w) # do the average denoise, wn = D^(-1) * w
    if type == 'gph': D_ = np.diag(1 / np.sqrt(D)); w_normal = np.dot(np.dot(D_, w), D_) # do the graph denoise (Graph Laplacian), wn = D^(-1/2) * w * D^(-1/2)
    return w_normal # shape: (num_cells, num_cells), w_normal is symmetric matrix if type is 'gph' and w_normal is not symmetric matrix if type is 'ave'

def TransitionFields(w): # TransitionFields method
    # w: weight matrix (w.shape=(num_cells, num_cells))
    zeroindex = np.argwhere(np.sum(w, axis=1) == 0).flatten() # zeroindex: cells with no connections, shape=(num_cells,)
    w = NE_dn(w, 'ave') # do the average denoise, shape: (num_cells, num_cells)
    w = w / np.sqrt(np.sum(np.abs(w), axis=0) + 1e-20).reshape(1, -1) # shape: (num_cells, num_cells), column-wise normalization
    w = np.dot(w, w.T) # W: transition matrix, shape: (num_cells, num_cells)
    w[zeroindex, :] = 0; w[:, zeroindex] = 0 # set the connections of cells with no connections to be zero
    return w # transition matrix, shape: (num_cells, num_cells)
    
def Dominateset(aff_matrix, NR_OF_KNN): # Dominateset method
    # aff_matrix: affinity matrix (aff_matrix.shape=(num_cells, num_cells)); NR_OF_KNN: number of neighbors
    A, B = np.sort(aff_matrix, axis=1)[:, ::-1], np.argsort(aff_matrix, axis=1)[:, ::-1] # A: values, A.shape=(num_cells, num_cells), B: indices, B.shape=(num_cells, num_cells)
    ind = np.tile(np.arange(len(aff_matrix)).reshape(-1, 1), (1, NR_OF_KNN)) ; loc = B[:, :NR_OF_KNN] # shape: (num_cells, NR_OF_KNN)
    PNN_matrix = np.zeros(aff_matrix.shape); PNN_matrix[ind, loc] = A[:, :NR_OF_KNN]; PNN_matrix = (PNN_matrix + PNN_matrix.T) / 2
    return PNN_matrix # PNN_matrix: affinity matrix, remove the elements that are not the top NR_OF_KNN largest values in each row

def NetworkEnhancement(w_in, order, K, alpha):
    # Network Enhancement (NE) method. 
    # reference: https://www.nature.com/articles/s41467-018-05469-x, https://snap.stanford.edu/ne/, https://github.com/wangboyunze/Network_Enhancement
    # w_in: weight matrix, w_in.shape=(num_cells, num_cells); order: order determines the extent of diffusion; K: number of neighbors; alpha: regularization parameter
    w_temp = w_in * (1 - np.eye(len(w_in))) # weight matrix, shape=(num_cells, num_cells), remove the diagonal elements
    nonzeroindex = np.argwhere(np.sum(w_temp, axis=1) > 0).flatten() # zeroindex: cells with no connections, shape=(num_cells,)
    w_nonzero = w_in[nonzeroindex, :][:, nonzeroindex] # remove the cells with no connections
    DD = np.sum(np.abs(w_nonzero), axis=0) # DD: degree matrix, DD.shape=(num_cells_nonzero,)
    w_nonzero = NE_dn(w_nonzero, type='ave'); w_nonzero = (w_nonzero + w_nonzero.T) / 2
    P = (Dominateset(np.abs(w_nonzero), min(K, len(w_nonzero) - 1))) * np.sign(w_nonzero)
    P = P + np.eye(len(P)) + np.diag(np.sum(np.abs(P.T), axis=0))
    P = TransitionFields(P); D, U = np.linalg.eig(P) # D: eigenvalues, D.shape=(num_cells_nonzero,), U: eigenvectors, U.shape=(num_cells_nonzero, num_cells_nonzero)
    d = D - 1e-20; d = (1 - alpha) * d / (1 - alpha * d ** order); D = np.diag(d)
    w_nonzero = np.dot(np.dot(U, D), U.T); w_nonzero = (w_nonzero * (1 - np.eye(len(w_nonzero)))) / (1 - np.diag(w_nonzero))
    D = np.diag(DD); w_nonzero =np.dot(D, w_nonzero); w_nonzero[w_nonzero < 0] = 0; w_nonzero = (w_nonzero + w_nonzero.T) / 2
    w_out = np.zeros((len(w_in), len(w_in))); 
    w_out[np.ix_(nonzeroindex, nonzeroindex)] = w_nonzero
    return w_out # shape: (num_cells, num_cells)

def getGraph(NE_path, features, K, method): # Construct a graph based on the cell features
    assert method in ['pearson', 'spearman', 'NE'], 'Possible methods: pearson, spearman, NE'
    if method == 'pearson': co_matrix = np.corrcoef(features) # shape: (num_cells, num_cells), Calculate the Pearson correlation coefficient between cells
    if method == 'spearman': co_matrix, _ = spearmanr(features.T) # shape: (num_cells, num_cells), Calculate the Spearman correlation
    if method == 'NE': # Construct the cell graph based on the NE (Network Enhancement) method
        if os.path.exists(NE_path):
            NE_matrix = pd.read_csv(NE_path).values
        else:
            w = np.corrcoef(features) # shape: (num_cells, num_cells)
            w = (w + 1) / 2 # scale the weight matrix to [0, 1]
            NE_matrix = NetworkEnhancement(w, order=3, K=min(20, len(w) // 10), alpha=0.9) # shape: (num_cells, num_cells)
            pd.DataFrame(NE_matrix).to_csv(NE_path, index=False)
        NE_matrix = NE_matrix - np.diag(np.diag(NE_matrix)) + np.diag(np.max(NE_matrix, axis=1)) # set the matrix diagonal to be the maximum value of the corresponding row
        data = NE_matrix.reshape(-1); data = np.sort(data); data = data[:-int(len(data)*0.02)]; # remove the top 2% of the data
        min_sh = data[0]; max_sh = data[-1]; delta = (max_sh - min_sh) / 100; temp_cnt = []; candi_sh = -1
        for i in range(20):
            s_sh = min_sh + delta * i; e_sh = s_sh + delta; temp_data = data[data > s_sh]; temp_data = temp_data[temp_data < e_sh]; temp_cnt.append([(s_sh + e_sh)/2, len(temp_data)])
        for i in range(len(temp_cnt)):
            pear_sh, pear_cnt = temp_cnt[i]
            if 0 < i < len(temp_cnt) - 1 and pear_cnt < temp_cnt[i+1][1] and pear_cnt < temp_cnt[i-1][1]:
                candi_sh = pear_sh; break
        if candi_sh < 0:
            for i in range(1, len(temp_cnt)):
                pear_sh, pear_cnt = temp_cnt[i]; candi_sh = pear_sh if pear_cnt * 2 < temp_cnt[i-1][1] else candi_sh
        if candi_sh == -1: candi_sh = 0.3
        propor = len(NE_matrix[NE_matrix <= candi_sh])/(len(NE_matrix)**2); thres = np.sort(NE_matrix)[:, int(len(NE_matrix) * propor)]; # shape: (num_cells,)
        co_matrix = np.corrcoef(features); co_matrix[NE_matrix < np.reshape(thres, (-1, 1))] = 0 # shape: (num_cells, num_cells), use the NE matrix to filter the co_matrix
    adj_K = np.zeros(co_matrix.shape) # shape: (num_cells, num_cells), make the upper K-th largest value of each row to be 1 and the others to be 0
    adj_K[co_matrix >= np.reshape(np.sort(co_matrix)[:,-K], (-1, 1))] = 1; adj_K[co_matrix < np.reshape(np.sort(co_matrix)[:,-K], (-1, 1))] = 0
    return adj_K # shape: (num_cells, num_cells), which every row has K non-zero values and the other values are zero

def load_data(data_path, NE_path, PCA_dim, n_clusters=20, is_NE=True):
    data = pd.read_csv(data_path, index_col=0, sep='\t'); cells = data.columns.values; genes = data.index.values
    features = data.values.T.copy() # [num_cells, num_genes] after transpose
    features = features / np.sum(features, axis=1).reshape(-1, 1) * 100000 # normalize total UMI counts to 100000 for each cell
    features = np.log2(features + 1) # log2 transformation
    K = len(cells) // (n_clusters * 10); K = min(K, 20); K = max(K, 6) # K is the number of neighbors to construct the cell graph, which is set to be 6 ~ 20
    adj = getGraph(NE_path, features, K, 'NE' if is_NE else 'pearson') # Construct the cell graph, shape: (num_cells, num_cells)
    if features.shape[0] > PCA_dim and features.shape[1] > PCA_dim:
        pca = PCA(n_components = PCA_dim); pca.fit(features); features = pca.transform(features) # shape: (num_cells, PCA_dim)
        # components = pca.components_ # shape: (PCA_dim, num_genes), meaning the original genes' weights on the PCA_dim dimensions
        # features = np.dot(features, components.T) # shape: (num_cells, PCA_dim)
    else:
        var = np.var(features, axis=0); min_var = np.sort(var)[-1 * PCA_dim]
        features = features[:, var >= min_var][:, :PCA_dim] # Select the top PCA_dim genes with the highest variance
    features = (features - np.mean(features)) / (np.std(features)) # shape: (num_cells, PCA_dim), normalize the features across all dimensions
    return adj, features, cells, genes # shape: (num_cells, num_cells), (num_cells, PCA_dim), (num_cells,), (num_genes,)

############################################################################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='name of dataset')
    parser.add_argument('n_clusters', type=int, help='expected number of clusters')
    parser.add_argument('--subtype_path', default=None, type=str, help='path of true labels for evaluation of ARI and NMI')
    parser.add_argument('--is_NE', default=True, type=bool, help='use NE denoise the cell graph or not')
    parser.add_argument('--PCA_dim', default=512, type=int, help='dimensionality of input feature matrix that transformed by PCA')
    parser.add_argument('--hidden_dim_1', default=64, type=int, help='number of neurons in the 1-st layer of encoder')
    parser.add_argument('--hidden_dim_2', default=16, type=int, help='number of neurons in the 2-nd layer of encoder')
    parser.add_argument('--n_attn_heads', default=4, type=int, help='number of heads for attention')
    parser.add_argument('--dropout_rate', default=0.4, type=float, help='dropout rate of neurons in autoencoder')
    parser.add_argument('--l2_reg', default=0, type=float, help='coefficient for L2 regularizition')
    parser.add_argument('--pre_lr', default=2e-4, type=float, help='learning rate for pre-training')
    parser.add_argument('--pre_epochs', default=200, type=int, help='number of epochs for pre-training')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate for training')
    parser.add_argument('--epochs', default=5000, type=int, help='number of epochs for training')
    parser.add_argument('--c1', default=1, type=float, help='weight of reconstruction loss')
    parser.add_argument('--c2', default=1, type=float, help='weight of clustering loss')
    args = parser.parse_args()
    os.makedirs(os.path.join('result', args.dataset), exist_ok=True)
    
    # 1\ Preprocess data
    print('Preprocess data')
    A, X, cells, genes = load_data(os.path.join('data', args.dataset, 'data.tsv'), os.path.join('data', args.dataset, 'NE.csv'), args.PCA_dim, args.n_clusters, args.is_NE)
    
    # 2\ Pre-train GAT autoencoder model
    print('Pre-train GAT autoencoder model')
    N = X.shape[0]; F = X.shape[1] # Number of cells and number of features
    X_in = Input(shape=(F,)); A_in = Input(shape=(N,))
    dropout_rate = 0. if args.k == 1 else args.dropout_rate # To avoid absurd results
    dropout1 = Dropout(dropout_rate)(X_in); graph_attention_1 = GraphAttention(args.hidden_dim_1, attn_heads=args.n_attn_heads, attn_heads_reduction='concat', dropout_rate=dropout_rate, activation='elu', kernel_regularizer=l1(args.l2_reg), attn_kernel_regularizer=l1(args.l2_reg))([dropout1, A_in])
    dropout2 = Dropout(dropout_rate)(graph_attention_1); graph_attention_2 = GraphAttention(args.hidden_dim_2, attn_heads=args.n_attn_heads, attn_heads_reduction='concat', dropout_rate=dropout_rate, activation='elu', kernel_regularizer=l1(args.l2_reg), attn_kernel_regularizer=l1(args.l2_reg))([dropout2, A_in])
    dropout3 = Dropout(dropout_rate)(graph_attention_2); graph_attention_3 = GraphAttention(args.hidden_dim_1, attn_heads=args.n_attn_heads, attn_heads_reduction='concat', dropout_rate=dropout_rate, activation='elu', kernel_regularizer=l1(args.l2_reg), attn_kernel_regularizer=l1(args.l2_reg))([dropout3, A_in])
    dropout4 = Dropout(dropout_rate)(graph_attention_3); graph_attention_4 = GraphAttention(F, attn_heads=args.n_attn_heads, attn_heads_reduction='average', dropout_rate=dropout_rate, activation='elu', kernel_regularizer=l1(args.l2_reg), attn_kernel_regularizer=l1(args.l2_reg))([dropout4, A_in])
    GAT_autoencoder = Model(inputs=[X_in, A_in], outputs=graph_attention_4)
    optimizer = Adam(lr=args.pre_lr)
    mae_class_loss = lambda y_true, y_pred: K.mean(K.obs(y_true - y_pred)) # mean absolute error loss, make the hidden representation of cells similar to the original cell features
    graph_recon_loss = lambda y_true, y_pred: K.mean(K.exp(-1 * A * K.sigmoid(K.dot(y_pred, K.transpose(y_pred))))) # make the graph structure of the hidden representation similar to the original graph structure
    GAT_autoencoder.compile(optimizer=optimizer, loss=mae_class_loss)
    GAT_autoencoder.summary()
    es_callback = EarlyStopping(monitor='loss', min_delta=0.1, patience=50)
    tb_callback = TensorBoard(batch_size=N)
    mc_callback = ModelCheckpoint(os.path.join('result', args.dataset, 'model_pretrain.h5'), monitor='loss', save_best_only=True, save_weights_only=True)
    GAT_autoencoder.fit([X, A], X, epochs=args.pre_epochs, batch_size=N, verbose=0, shuffle=False, callbacks=[es_callback, tb_callback, mc_callback]) # Train the model
    
    # 3\ Train the total model
    print('Train the total model')
    hidden_model = Model(inputs=[X_in, A_in], outputs=graph_attention_2)
    hidden = hidden_model.predict([X, A], batch_size=N).astype(float)
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(hidden) # y_pred = kmeans.predict(hidden) # Get the predicted labels is equivalent to kmeans.labels_
    y_pred = kmeans.labels_; pre_centers = kmeans.cluster_centers_ # Get k-means clustering results of hidden representation of cells
    y_pred_last = np.copy(y_pred) # Initialize the predicted labels
    soft_cluster_layer = ClusteringLayer(n_clusters=args.n_clusters, clusters=pre_centers, mode='q', threshold=0, name='clustering')(dropout3) # Add the soft_clustering layer
    model = Model(inputs=[X_in, A_in], outputs=[graph_attention_4, soft_cluster_layer, graph_attention_2])
    optimizer = Adam(lr=args.lr)
    pred_loss = lambda y_true, y_pred: y_pred # dummy loss function for the clustering layer, which is not used in training
    model.compile(optimizer=optimizer, loss=[mae_class_loss, 'kld', pred_loss], loss_weights=[args.c1, args.c2, 0])
    sil_logs = []; max_sil = 0; loss = 0; final_pred = None; final_hidden = None
    for ite in range(args.epochs + 1):
        if ite % 2 == 0:
            recon, q, hidden = model.predict([X, A], batch_size=N, verbose=0) # shape: (N x F), (N x n_clusters), (N x F)
            y_pred = q.argmax(1) # Get the predicted labels
            p = q ** 2 / q.sum(0); p = (p.T / p.sum(1)).T # shape: (N x n_clusters), Calculate the P matrix as the target distribution 
            sil_hidden = metrics.silhouette_score(hidden, y_pred, metric='euclidean'); sil_logs.append(sil_hidden)
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]; y_pred_last = np.copy(y_pred)
            print('Iter:', ite, ', sil_hid:', np.round(sil_hidden, 3), ', delta_label', np.round(delta_label, 3), ', loss:', np.round(loss, 2))
            if sil_hidden >= max_sil: 
                final_pred = y_pred; final_hidden = hidden; max_sil = sil_hidden
            if len(sil_logs) >= 3:
                if sil_logs[-1] - sil_logs[-2] <= -0.05: break
            if len(sil_logs) >= 30 * 2:
                if np.mean(sil_logs[-30:]) - np.mean(sil_logs[-60: -30]) <= 0.02: break
        loss = model.train_on_batch(x=[X, A], y=[X, p, hidden])
    
    # 4\ Save results and evaluate the performance
    print('Save results')
    result = np.array([[cells[i], final_pred[i]] for i in range(len(cells))])
    pd.DataFrame(result, columns=['cell', 'label']).to_csv(os.path.join('result', args.dataset, 'result.txt'), sep='\t', index=False)
    hidden = final_hidden.astype(float)
    pd.DataFrame(hidden).to_csv(os.path.join('result', args.dataset, 'hidden.tsv'), sep='\t')
    if args.subtype_path:
        cell_subtype = pd.read_csv(args.subtype_path, sep='\t').values[:, -1].astype(int)
        ARI = metrics.adjusted_rand_score(cell_subtype, final_pred); NMI = metrics.normalized_mutual_info_score(cell_subtype, final_pred)
        print('ARI: {}'.format(ARI), 'NMI: {}'.format(NMI))
        