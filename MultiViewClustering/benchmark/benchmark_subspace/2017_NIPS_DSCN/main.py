import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
from munkres import Munkres
import matplotlib.pyplot as plt

def load_yaleb(data_path):
    """Load YaleB from .mat; return Img [N, H, W, 1], Label [N]."""
    data = sio.loadmat(data_path)
    img = data['Y']
    I = []
    Label = []
    for i in range(img.shape[2]):
        for j in range(img.shape[1]):
            temp = np.reshape(img[:, j, i], [42, 48])
            Label.append(i)
            I.append(temp)
    I = np.array(I)
    Label = np.array(Label[:])
    Img = np.transpose(I, [0, 2, 1])
    Img = np.expand_dims(Img[:], 3)
    return Img.astype(np.float32), np.squeeze(np.array(Label))

def next_batch(data, _index_in_epoch, batch_size, _epochs_completed):
    _num_examples = data.shape[0]
    start = _index_in_epoch
    _index_in_epoch += batch_size
    if _index_in_epoch > _num_examples:
        _epochs_completed += 1
        perm = np.arange(_num_examples)
        np.random.shuffle(perm)
        data = data[perm]
        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= _num_examples
    end = _index_in_epoch
    return data[start:end], _index_in_epoch, _epochs_completed

def best_map(L1, L2):
    # L1: groundtruth labels, L2: clustering labels
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = (L1 == Label1[i]).astype(float)
        for j in range(nClass2):
            ind_cla2 = (L2 == Label2[j]).astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while not stop:
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C
    return Cp

def build_aff(C):
    N = C.shape[0]
    Cabs = np.abs(C)
    ind = np.argsort(-Cabs, 0)
    for i in range(N):
        Cabs[:, i] = Cabs[:, i] / (Cabs[ind[0, i], i] + 1e-6)
    Cksym = Cabs + Cabs.T
    return Cksym

def post_proC(C, K, d, alpha):
    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, min(r, C.shape[0] - 1), v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / (L.max() + 1e-8)
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed', assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L

def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate

def build_laplacian(C):
    C = 0.5 * (np.abs(C) + np.abs(C.T))
    W = np.sum(C, axis=0)
    W = np.diag(1.0 / (W + 1e-8))
    L = W.dot(C)
    return L

# --------------- ConvAE Pre-train (encoder + decoder only) ---------------
class ConvAEPretrain(nn.Module):
    """Encoder + decoder for pre-training (no Coef)."""
    def __init__(self, n_input, kernel_size, n_hidden, batch_size=256):
        super().__init__()
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        # Encoder: same padding for stride 2 -> ceil(in/2). k=5 -> pad 2, k=3 -> pad 1
        self.enc_conv0 = nn.Conv2d(1, n_hidden[0], kernel_size[0], stride=2, padding=2)
        self.enc_conv1 = nn.Conv2d(n_hidden[0], n_hidden[1], kernel_size[1], stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(n_hidden[1], n_hidden[2], kernel_size[2], stride=2, padding=1)
        # Decoder: mirror; we use output_padding to match TF shapes (6->12, 12->24, 24->48 and 6->11, 11->21, 21->42)
        self.dec_conv0 = nn.ConvTranspose2d(n_hidden[2], n_hidden[1], kernel_size[2], stride=2, padding=1, output_padding=(1, 0))
        self.dec_conv1 = nn.ConvTranspose2d(n_hidden[1], n_hidden[0], kernel_size[1], stride=2, padding=1, output_padding=(1, 1))
        self.dec_conv2 = nn.ConvTranspose2d(n_hidden[0], 1, kernel_size[0], stride=2, padding=2, output_padding=(1, 1))
        self._init_weights()

    def _init_weights(self):
        for m in (self.enc_conv0, self.enc_conv1, self.enc_conv2,
                  self.dec_conv0, self.dec_conv1, self.dec_conv2):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encoder(self, x):
        shapes = []
        h = x
        shapes.append(list(h.shape))
        h = torch.relu(self.enc_conv0(h))
        shapes.append(list(h.shape))
        h = torch.relu(self.enc_conv1(h))
        shapes.append(list(h.shape))
        h = torch.relu(self.enc_conv2(h))
        return h, shapes

    def decoder(self, z, shapes):
        h = torch.relu(self.dec_conv0(z))
        h = torch.relu(self.dec_conv1(h))
        h = torch.relu(self.dec_conv2(h))
        return h

    def forward(self, x, denoise=False):
        if denoise:
            x = x + torch.randn_like(x, device=x.device) * 0.2
        latent, shapes = self.encoder(x)
        z = latent.reshape(latent.size(0), -1)
        x_r = self.decoder(latent, shapes)
        return x_r, z

    def train_step(self, x, optimizer, device):
        self.train()
        optimizer.zero_grad()
        x = x.to(device)
        x_r, _ = self.forward(x, denoise=False)
        loss = 0.5 * torch.sum((x_r - x) ** 2)
        loss.backward()
        optimizer.step()
        return loss.item()

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print("model saved in file: %s" % path)

    def load_model(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device), strict=True)
        print("model loaded from %s" % path)


# --------------- ConvAE DSC (encoder + Coef + decoder) ---------------
class ConvAEDSC(nn.Module):
    """Encoder + Coef + decoder for fine-tuning and clustering."""
    def __init__(self, n_input, kernel_size, n_hidden, batch_size, reg_constant1=1.0, re_constant2=1.0):
        super().__init__()
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.reg_constant1 = reg_constant1
        self.re_constant2 = re_constant2
        self.enc_conv0 = nn.Conv2d(1, n_hidden[0], kernel_size[0], stride=2, padding=2)
        self.enc_conv1 = nn.Conv2d(n_hidden[0], n_hidden[1], kernel_size[1], stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(n_hidden[1], n_hidden[2], kernel_size[2], stride=2, padding=1)
        self.dec_conv0 = nn.ConvTranspose2d(n_hidden[2], n_hidden[1], kernel_size[2], stride=2, padding=1, output_padding=(1, 0))
        self.dec_conv1 = nn.ConvTranspose2d(n_hidden[1], n_hidden[0], kernel_size[1], stride=2, padding=1, output_padding=(1, 1))
        self.dec_conv2 = nn.ConvTranspose2d(n_hidden[0], 1, kernel_size[0], stride=2, padding=2, output_padding=(1, 1))
        self.Coef = nn.Parameter(torch.ones(batch_size, batch_size, dtype=torch.float32) * 1e-4)
        self._init_weights()

    def _init_weights(self):
        for m in (self.enc_conv0, self.enc_conv1, self.enc_conv2,
                  self.dec_conv0, self.dec_conv1, self.dec_conv2):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encoder(self, x):
        h = torch.relu(self.enc_conv0(x))
        h = torch.relu(self.enc_conv1(h))
        h = torch.relu(self.enc_conv2(h))
        return h

    def decoder(self, z):
        h = torch.relu(self.dec_conv0(z))
        h = torch.relu(self.dec_conv1(h))
        h = torch.relu(self.dec_conv2(h))
        return h

    def forward(self, x, denoise=False):
        if denoise:
            x = x + torch.randn_like(x, device=x.device) * 0.2
        latent = self.encoder(x)
        z = latent.reshape(latent.size(0), -1)
        z_c = torch.mm(self.Coef, z)
        latent_c = z_c.reshape(latent.shape)
        x_r = self.decoder(latent_c)
        return x_r, z, z_c

    def loss_fn(self, x, x_r, z, z_c):
        reconst = 0.5 * torch.sum((x_r - x) ** 2)
        reg = torch.sum(self.Coef ** 2)
        selfexpress = 0.5 * torch.sum((z_c - z) ** 2)
        return reconst + self.reg_constant1 * reg + self.re_constant2 * selfexpress

    def load_pretrained_encoder_decoder(self, path, device):
        state = torch.load(path, map_location=device)
        # Load only encoder/decoder keys; skip Coef
        own = self.state_dict()
        loaded = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        self.load_state_dict(loaded, strict=False)
        print("pretrained encoder/decoder loaded from %s (strict=False)" % path)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print("model saved in file: %s" % path)

# --------------- Pre-train and eval flows ---------------
def train_face(Img, model, n_input, batch_size, device, lr=1e-3, max_iters=None,
               save_path=None, display_step=300, save_step=900):
    """Pre-train ConvAEPretrain on YaleB with random batches."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    _index_in_epoch = 0
    _epochs = 0
    it = 0
    while True:
        batch_x, _index_in_epoch, _epochs = next_batch(Img, _index_in_epoch, batch_size, _epochs)
        # [B, H, W, 1] -> [B, 1, H, W]
        batch_x = np.reshape(batch_x, [batch_size, n_input[0], n_input[1], 1])
        batch_x = torch.from_numpy(batch_x).permute(0, 3, 1, 2).to(device)
        cost = model.train_step(batch_x, optimizer, device)
        it += 1
        if max_iters is not None and it >= max_iters:
            break
        if it % display_step == 0:
            print("epoch: %d" % _epochs)
            print("cost: %.8f" % (cost / batch_size))
        if save_path and save_step and it % save_step == 0:
            model.save_model(save_path)
    if save_path:
        model.save_model(save_path)

def test_face_reconstruct(Img, model, n_input, device, num_samples=100):
    """Visualize reconstructions (optional, after pretrain)."""
    model.eval()
    batch_x = Img[200:200 + num_samples]
    batch_x = np.reshape(batch_x, [num_samples, n_input[0], n_input[1], 1])
    x_t = torch.from_numpy(batch_x).float().permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        x_r, _ = model(x_t)
    x_r = x_r.cpu().numpy().transpose(0, 2, 3, 1)
    batch_x = batch_x
    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(batch_x[i, :, :, 0], vmin=0, vmax=255, cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(x_r[i, :, :, 0], vmin=0, vmax=255, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()
    plt.show()

def test_face(Img, Label, num_class, device, n_input, kernel_size, n_hidden,
              restore_path, reg1=1.0, reg2_mult=None, max_step=None, display_step=None):
    """
    Run DSC fine-tune + clustering for one num_class over all experiments (0 .. 39-num_class).
    Returns (mean_acc, median_acc).
    """
    if reg2_mult is None:
        reg2_mult = 1.0 * (10 ** (num_class / 10.0 - 3.0))
    batch_size = num_class * 64
    if max_step is None:
        max_step = 50 + num_class * 25
    if display_step is None:
        display_step = max_step
    alpha = max(0.4 - (num_class - 1) / 10 * 0.1, 0.1)
    print(alpha)
    acc_list = []
    n_exp = 39 - num_class
    for i in range(n_exp):
        face_batch = np.array(Img[64 * i:64 * (i + num_class), :], dtype=np.float32)
        label_batch = np.array(Label[64 * i:64 * (i + num_class)])
        label_batch = label_batch - label_batch.min() + 1
        label_batch = np.squeeze(label_batch)
        model = ConvAEDSC(n_input=n_input, kernel_size=kernel_size, n_hidden=n_hidden, batch_size=batch_size, reg_constant1=reg1, re_constant2=reg2_mult)
        model.load_pretrained_encoder_decoder(restore_path, device)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        x_t = np.reshape(face_batch, [batch_size, n_input[0], n_input[1], 1])
        x_t = torch.from_numpy(x_t).float().permute(0, 3, 1, 2).to(device)
        for epoch in range(1, max_step + 1):
            model.train()
            optimizer.zero_grad()
            x_r, z, z_c = model(x_t, denoise=False)
            loss = model.loss_fn(x_t, x_r, z, z_c)
            loss.backward()
            optimizer.step()
            if epoch % display_step == 0:
                cost_val = 0.5 * torch.sum((x_r - x_t) ** 2).item()
                print("epoch: %d" % epoch, "cost: %.8f" % (cost_val / batch_size))
        # One accuracy per experiment (after fine-tune)
        Coef = model.Coef.detach().cpu().numpy()
        Coef = thrC(Coef, alpha)
        y_x, _ = post_proC(Coef, int(label_batch.max()), 10, 3.5)
        missrate_x = err_rate(label_batch, y_x)
        acc_x = 1 - missrate_x
        print("experiment: %d" % i, "our accuracy: %.4f" % acc_x)
        acc_list.append(acc_x)
    acc_arr = np.array(acc_list)
    m = np.mean(acc_arr)
    me = np.median(acc_arr)
    print("%d subjects:" % num_class)
    print("Mean: %.4f%%" % ((1 - m) * 100))
    print("Median: %.4f%%" % ((1 - me) * 100))
    print(acc_arr)
    return 1 - m, 1 - me

def ae_feature_clustering(model, X, save_path='AE_YaleB.mat', device=None):
    """Extract latent Z and save to .mat (optional)."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'encoder'):
            # ConvAEDSC
            latent = model.encoder(X)
            Z = latent.reshape(latent.size(0), -1).cpu().numpy()
        else:
            _, Z = model(X)
            Z = Z.cpu().numpy()
    sio.savemat(save_path, dict(Z=Z))
    print("Z saved to %s" % save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSC-Net EYaleB (PyTorch): pretrain and/or evaluate')
    parser.add_argument('--data_path', type=str, default='./data/YaleBCrop025.mat', help='Path to YaleBCrop025.mat')
    parser.add_argument('--ckpt_dir', type=str, default='./result/ckpt', help='Directory for saving/loading pretrained checkpoint')
    parser.add_argument('--mode', type=str, choices=['pretrain', 'eval', 'both'], default='both', help='pretrain only, eval only (need existing ckpt), or both')
    parser.add_argument('--pretrain_iters', type=int, default=999999, help='Max iterations for pretrain (default: run until interrupted)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--visualize', action='store_true', help='After pretrain, show reconstruction comparison (requires matplotlib)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    data_path = args.data_path
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    n_input = [48, 42]
    kernel_size = [5, 3, 3]
    n_hidden = [10, 20, 30]
    all_subjects = [10, 15, 20, 25, 30, 35, 38]
    if args.mode in ('pretrain', 'both'):
        if not os.path.isfile(data_path):
            print("Data not found at %s; skipping pretrain. Run with --mode eval and provide pretrained ckpt." % data_path)
        else:
            Img, Label = load_yaleb(data_path)
            batch_size = Img.shape[0]
            model = ConvAEPretrain(n_input=n_input, kernel_size=kernel_size, n_hidden=n_hidden, batch_size=batch_size)
            save_path = ckpt_dir + '/model-102030-48x42-yaleb.pt'
            train_face(Img, model, n_input, batch_size, device, lr=1.0e-3, max_iters=args.pretrain_iters,
                      save_path=save_path, display_step=300, save_step=900)
            if args.visualize:
                test_face_reconstruct(Img, model, n_input, device)
    if args.mode in ('eval', 'both'):
        restore_path = ckpt_dir + '/model-102030-48x42-yaleb.pt'
        if not os.path.isfile(restore_path):
            print("Pretrained checkpoint not found at %s; run pretrain first or set --mode pretrain" % restore_path)
        else:
            if args.mode == 'eval' and not os.path.isfile(data_path):
                print("Data path required for eval")
            else:
                Img, Label = load_yaleb(data_path)
                avg_list, med_list = [], []
                for num_class in all_subjects:
                    avg_i, med_i = test_face(Img, Label, num_class, device, n_input, kernel_size, n_hidden, restore_path)
                    avg_list.append(avg_i)
                    med_list.append(med_i)
                for i, num_class in enumerate(all_subjects):
                    print("%d subjects:" % num_class)
                    print("Mean: %.4f%%" % (avg_list[i] * 100), "Median: %.4f%%" % (med_list[i] * 100))
