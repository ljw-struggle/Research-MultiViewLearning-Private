import os, random, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torchvision
from torchvision.utils import save_image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(nn.Linear(784, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, 2000), nn.ReLU()); self.fc1 = nn.Linear(2000, 10); self.fc2 = nn.Linear(2000, 10)
        self.decoder = torch.nn.Sequential(nn.Linear(10, 2000), nn.ReLU(), nn.Linear(2000, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, 784), nn.Sigmoid())

    def forward(self, x):
        h = self.encoder(x.view(-1, 784)); mu = self.fc1(h); logvar = self.fc2(h); std = (0.5 * logvar).exp(); 
        z = torch.randn_like(std) * std + mu  # reparameterization trick, torch.randn_like(std) generates random noise with the same shape as std using the normal distribution
        x_recon = self.decoder(z) # decode the latent variable z to reconstruct the input x
        return x_recon, mu, logvar  # return reconstructed x, mu, and logvar

    @staticmethod
    def loss_func(x_recon, x, mu, logvar):
        BCE = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='none').sum(dim=1) # shape: [batch_size]
        # MSE = F.mse_loss(x_recon, x.view(-1, 784), reduction='none').sum(dim=1) # shape: [batch_size]
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) # shape: [batch_size]
        return torch.mean(BCE + KLD)  # return the average loss over the batch

class AE(torch.nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(784, 500), torch.nn.ReLU(), torch.nn.Linear(500, 500), torch.nn.ReLU(), torch.nn.Linear(500, 2000), torch.nn.ReLU(), torch.nn.Linear(2000, 10))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(10, 2000), torch.nn.ReLU(), torch.nn.Linear(2000, 500), torch.nn.ReLU(), torch.nn.Linear(500, 500), torch.nn.ReLU(), torch.nn.Linear(500, 784), torch.nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x.view(-1, 784))
        x_recon = self.decoder(z)
        return x_recon, z
    
    @staticmethod
    def loss_func(x_recon, x):
        BCE = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='none').sum(dim=1)
        # MSE = F.mse_loss(x_recon, x.view(-1, 784), reduction='none').sum(dim=1) # shape: [batch_size]
        return torch.mean(BCE)  # return the average loss over the batch

def clustering_acc(y_true, y_pred): # y_pred and y_true are numpy arrays
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.array([[sum((y_pred == i) & (y_true == j)) for j in range(D)] for i in range(D)], dtype=np.int64)
    # w = np.zeros((D, D), dtype=np.int64)
    # for i in range(y_pred.size):
    #     w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w) # align clusters using the Hungarian algorithm
    return sum([w[i][j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.shape[0] 

def clustering_loss(embedding, cluster_centers):
    alpha = 1.0; q = 1.0 / ((1.0 + torch.sum((embedding.unsqueeze(1) - cluster_centers) ** 2, dim=2) / alpha) ** ((alpha + 1.0) / 2.0)); q = q / torch.sum(q, dim=1, keepdim=True) # similarity matrix q using t-distribution, shape: [batch_size, n_clusters]
    p = q ** 2 / torch.sum(q, dim=0); p = p / torch.sum(p, dim=1, keepdim=True).detach() # target distribution p, detach p to avoid computing gradients for p, shape: [batch_size, n_clusters]
    loss = torch.mean(torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10)), dim=1)) # numerical stability with log(0) handling
    # loss = F.kl_div(q.log(), p, reduction='batchmean')  # Kullback-Leibler divergence loss
    return loss, p
    
if __name__ == '__main__':
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    os.makedirs('./result', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ## Stage 1: Pre-train the VAE
    train_data = torchvision.datasets.MNIST(root='./mnist', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    test_data = torchvision.datasets.MNIST(root='./mnist', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=0)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=128, shuffle=False, num_workers=0)
    vae = VAE().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(vae.parameters(), lr=1.0, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True)
    for epoch in range(300):
        vae.train()
        train_loss = 0
        for batch_idx, (data, label_train) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = vae(data)
            loss = vae.loss_func(x_recon, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item() * len(data)
            print('====> PreTrain Epoch: {} [{}/{} ({:.0f}%)]'.format(epoch, batch_idx * 128 + len(data), len(train_loader.dataset), 100. * (batch_idx+1) / len(train_loader))) if batch_idx % 100 == 0 else None
        avg_loss = train_loss / len(train_loader.dataset)
        print('====> PreTrain Epoch: {} Train Loss: {:.4f}'.format(epoch, avg_loss))
        vae.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_dix, (data, label_test) in enumerate(test_loader):
                data = data.to(device)
                x_recon, mu, logvar = vae(data)
                loss = vae.loss_func(x_recon, data, mu, logvar)
                test_loss = test_loss + loss.item() * len(data)
                if batch_dix == 0:
                    comparison = torch.cat([data[:8], x_recon.view(128, 1, 28, 28)[:8]])
                    save_image(comparison.cpu(), './result/reconstruction_' + str(epoch) + '.png', nrow=8, normalize=True)
        test_loss = test_loss / len(test_loader.dataset)
        print('====> PreTrain Epoch: Test Loss: {:.4f}'.format(test_loss))
        scheduler.step(test_loss)
        torch.save(vae.state_dict(), './result/pretrain_vae.pt')
        
    ## Stage 2: Fine-tune the VAE with DEC
    vae = VAE().to(device)
    vae.load_state_dict(torch.load('./result/pretrain_vae.pt'))
    train_data = torchvision.datasets.MNIST(root='./mnist', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    data = (train_data.data.type(torch.FloatTensor).view(-1, 784) / 255.0).to(device)  # normalize the data to [0, 1], is applied by torchvision.transforms.ToTensor(), but train_data.data is in uint8 format so we need to convert it manually
    label = train_data.targets.type(torch.LongTensor).to(device)
    x_recon, mu, logvar = vae(data)
    kmeans = KMeans(n_clusters=10, n_init=20)
    cluster_predict = kmeans.fit_predict(mu.detach().cpu().numpy())
    cluster_centers = torch.from_numpy(kmeans.cluster_centers_).float().to(device).requires_grad_(True)
    print('Pre-trained auto-encoder accuracy: {}'.format(clustering_acc(cluster_predict, label.cpu().numpy())))
    train_data = torchvision.datasets.MNIST('../mnist', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=2056, shuffle=True, num_workers=1)
    optimizer = torch.optim.SGD(list(vae.encoder.parameters()) + list(vae.fc1.parameters()) + [cluster_centers], lr=1e-3)
    for epoch in range(20):
        vae.train()
        print('====> Fine-tune Epoch: {}'.format(epoch))
        for step, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_x = batch_x.view(-1, 784).to(device)
            x_recon, mu, logvar = vae(batch_x)
            loss, p = clustering_loss(mu, cluster_centers)
            loss.backward()
            optimizer.step()
        vae.eval()
        with torch.no_grad():
            x_recon, mu, logvar = vae(data)
            loss, p = clustering_loss(mu, cluster_centers)
            pred_label = torch.argmax(p, dim=1)
            accuracy = clustering_acc(pred_label.cpu().numpy(), label.cpu().numpy())
            print('====> Epoch: {} Accuracy: {}'.format(epoch, accuracy))
            