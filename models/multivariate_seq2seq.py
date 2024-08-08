import torch
import torch.nn as nn
from torch import Tensor



class MultivariateSeq2SeqModel(nn.Module):

    def __init__(self, T: int, D: int, tau: int) -> None:
        super(MultivariateSeq2SeqModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)

        self.dense1 = nn.Linear(in_features=21, out_features=256)
        self.dense2 = nn.Linear(in_features=256, out_features=512)
        self.dense3 = nn.Linear(in_features=512, out_features=32)
        self.dense4 = nn.Linear(in_features=32, out_features=6)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=0)


    def forward(self, x_seq: Tensor) -> Tensor:
        # x_seq = x_seq.T
        # print(x_seq.shape)
        # input()
        h = self.conv1(x_seq)
        # print(h.shape)
        # input()
        h = self.pool(h)
        # print(h.shape)
        # input()
        h = self.conv2(h)
        # print(h.shape)
        # input()
        h = self.pool(h)
        # print(h.shape)
        # input()
        h = self.conv3(h)
        # print(h.shape)
        # input()
        h = self.pool(h)
        # print(h.shape)
        # input()
        h = h.view(h.size(0), -1)
        h = self.dense1(h)
        # print(h.shape)
        # input()
        h = self.dense2(h)
        # print(h.shape)
        # input()
        h = self.dense3(h)
        # print(h.shape)
        # input()
        h = self.dense4(h)
        # print(h.shape)
        # input()
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






























# class MultivariateSeq2SeqModel(nn.Module):
#     """A PyTorch neural network model for multivariate data.

#     Args:
#         T (int): The input dimension.
#         D (int): The hidden dimension.
#         tau (int): The output dimension.

#     Attributes:
#         fc1 (nn.Linear): The first fully connected layer.
#         fc2 (nn.Linear): The second fully connected layer.
#         relu (nn.ReLU): The ReLU activation function.

#     Methods:
#         encode(x: Tensor) -> Tensor: Encodes the input tensor using the first fully connected layer and ReLU activation.
#         query(h: Tensor) -> Tensor: Performs a query on the hidden state tensor using the second fully connected layer.
#         forward(x_seq: Tensor) -> Tensor: Performs forward propagation on the input sequence tensor.

#     """

#     def __init__(self, T: int, D: int, tau: int) -> None:
#         super(MultivariateSeq2SeqModel, self).__init__()
#         print("T : ", T)
#         print("D : ", D)
#         self.fc1 = nn.Linear(T, D)
#         self.fc2 = nn.Linear(D, tau)
#         self.relu = nn.ReLU()

#     def encode(self, x: Tensor) -> Tensor:
#         """Encodes the input tensor.

#         Args:
#             x (Tensor): The input tensor.

#         Returns:
#             Tensor: The encoded tensor.

#         """
#         h = self.fc1(x)
#         return self.relu(h)

#     def query(self, h: Tensor) -> Tensor:
#         """Performs a query on the hidden state tensor.

#         Args:
#             h (Tensor): The hidden state tensor.

#         Returns:
#             Tensor: The query result.

#         """
#         return self.fc2(h)

#     def binarize_encoder(self) -> None:
#         self.fc1.weight = torch.nn.Parameter( torch.sign(self.fc1.weight), requires_grad=False )

#     def forward(self, x_seq: Tensor) -> Tensor:
#         """Performs forward propagation on the input sequence tensor.

#         Args:
#             x_seq (Tensor): The input sequence tensor.

#         Returns:
#             Tensor: The output tensor.

#         """
       

#         h = self.encode(x_seq)
#         h = self.query(h)
#         return h
