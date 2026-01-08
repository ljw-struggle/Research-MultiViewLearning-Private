
import torch.nn as nn
from torch.nn.functional import normalize
import torch

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)
    

class Autoencoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        # 初始化Autoencoder类
        super(Autoencoder, self).__init__()
        # 初始化编码器
        self.encoder = Encoder(input_dim, feature_dim)
        # 初始化解码器
        self.decoder = Decoder(input_dim, feature_dim)

    def forward(self, x):
        print(x.shape)
        z = self.encoder(x)
        
        x_hat = self.decoder(z)
        return x_hat, z


 
