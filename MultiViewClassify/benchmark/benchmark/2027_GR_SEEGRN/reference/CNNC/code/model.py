# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class CNNC(nn.Module):
    def __init__(self):
        super(CNNC, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding='same'), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding='same'), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same'), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same'), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.5),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same'), nn.ReLU(), 
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.5)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * (32 // 8) * (32 // 8), 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        # x.shape = (batch_size, 32, 32, 1)
        x = x.permute(0, 3, 1, 2)  # x.shape = (batch_size, 1, 32, 32)
        conv_output = self.conv_layers(x) # conv_output.shape = (batch_size, 128, 4, 4)
        conv_output = conv_output.reshape(conv_output.size(0), -1)  # conv_output.shape = (batch_size, 128 * 4 * 4)
        fc_output = self.fc_layers(conv_output)
        return fc_output
