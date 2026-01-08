# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class DGRNS(nn.Module):
    def __init__(self):
        super(DGRNS, self).__init__()
        self.rnn = nn.GRU(8, 128, batch_first=True)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding='same'), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.25),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding='same'), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * (8 // 4) * (128 // 4), 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        cnn_input = rnn_output.unsqueeze(1)  # Add a channel dimension for the CNN
        cnn_output = self.cnn(cnn_input)
        cnn_output = cnn_output.view(cnn_output.size(0), -1)  # Flatten the CNN output
        output = self.fc(cnn_output)
        return output
