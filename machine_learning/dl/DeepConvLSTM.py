# ------------------------------------------------------------------------
# DeepConvLSTM model based on architecture suggested by Ordonez and Roggen 
# https://www.mdpi.com/1424-8220/16/1/115
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# Email: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------

from torch import nn
import torch


class DeepConvLSTM(nn.Module):
    def __init__(self, channels, classes, window_size, conv_kernels=64, conv_kernel_size=5, lstm_units=128,
                 lstm_layers=2, dropout=0.5):
        super(DeepConvLSTM, self).__init__()

        # Convolve over all axes (here: 6) with given kernel size
        self.conv1 = nn.Conv2d(1, conv_kernels, (conv_kernel_size, channels), padding="same")
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, channels), padding="same")
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, channels), padding="same")
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, channels), padding="same")
        self.lstm = nn.LSTM(channels * conv_kernels, lstm_units, num_layers=lstm_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_units, classes)
        self.activation = nn.ReLU()
        self.lstm_units = lstm_units
        self.classes = classes

    def forward(self, x):
        x = x.unsqueeze(1)  # Add dimension for conv2d layers - 1 "channel" input
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.permute([0, 2, 1, 3])

        x = x.reshape(x.shape[0], x.shape[1], -1)
        x, h = self.lstm(x)
        x = x[:, -1, :]
        x = x.view(-1, self.lstm_units)
        x = self.dropout(x)
        return self.classifier(x)