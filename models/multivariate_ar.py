import torch
import torch.nn as nn
from torch import Tensor


class MultivariateARModel(nn.Module):

    def __init__(self, T: int, D: int, tau: int) -> None:
        super(MultivariateARModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)

        self.dense1 = nn.Linear(in_features=21, out_features=256)
        self.dense2 = nn.Linear(in_features=256, out_features=512)
        self.dense3 = nn.Linear(in_features=512, out_features=32)
        self.dense4 = nn.Linear(in_features=32, out_features=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=0)


    def forward(self, x_seq: Tensor) -> Tensor:
        x_seq = x_seq.T
        print(x_seq.shape)
        input()
        h = self.conv1(x_seq)
        print(h.shape)
        input()
        h = self.pool(h)
        print(h.shape)
        input()
        h = self.conv2(h)
        print(h.shape)
        input()
        h = self.pool(h)
        print(h.shape)
        input()
        h = self.conv3(h)
        print(h.shape)
        input()
        h = self.pool(h)
        print(h.shape)
        input()
        h = h.view(h.size(0), -1)
        h = self.dense1(h)
        print(h.shape)
        input()
        h = self.dense2(h)
        print(h.shape)
        input()
        h = self.dense3(h)
        print(h.shape)
        input()
        h = self.dense4(h)
        print(h.shape)
        input()
        return h


# Convolution 1D (Batch size, 24, 8)
# Max pooling 1D (Batch size, 12, 8)
# Convolution 1D (Batch size, 12, 16)
# Max pooling 1D (Batch size, 6, 16)
# Convolution 1D (Batch size, 6, 32)
# Max pooling 1D (Batch size, 3, 32)
# LSTM (Batch size, 64)
# Dense (Batch size, 256)
# Dense (Batch size, 32)
# Dense (Batch size, 12)














# class MultivariateARModel(nn.Module):

#     def __init__(self, T: int, D: int, tau: int) -> None:
#         super(MultivariateARModel, self).__init__()
#         self.linear1 = nn.Linear(T, D)
#         self.linear2 = nn.Linear(D, 1)
#         self.relu = nn.ReLU()

#     def encode(self, x: Tensor) -> Tensor:
#         h = self.linear1(x.T)
#         return self.relu(h)


#     def binarize_encoder(self) -> None:
#         self.linear1.weight = torch.nn.Parameter( torch.sign(self.linear1.weight), requires_grad=False )

#     def query(self, h: Tensor) -> Tensor:
#         return self.linear2(h)

#     def forward(self, x_seq: Tensor) -> Tensor:
#         h = self.encode(x_seq)
#         h = self.query(h)
#         return h