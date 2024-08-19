import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from data.data_loader import Dataset_ohio

import matplotlib.pyplot as plt
from torch import Tensor
import torch.nn.functional as F
import torchhd
from torchhd import embeddings

warnings.filterwarnings("ignore")


def cumavg(m):
    cumsum = np.cumsum(m)
    return cumsum / np.arange(1, cumsum.size + 1)

def MAE(pred, true):
    return np.sum(np.absolute((true - pred))) / len(true)

def RMSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2) / len(true))

def CORR(pred, true):
    mean_x = np.mean(pred)
    mean_y = np.mean(true)
    numerator = np.sum((pred - mean_x) * (true - mean_y))
    if np.var(true) >0:
        denominator = np.sqrt(np.sum((pred - mean_x) ** 2) * np.sum((true - mean_y) ** 2))
    else:
        denominator = np.sqrt(np.sum((pred - mean_x) ** 2))
    r = numerator / denominator
    return r

def metric(pred, true):
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    return mae, rmse


class RegHD(nn.Module):
    def __init__(self, SIZE, DIMENSIONS, device):
        super(RegHD, self).__init__()
        self.lr = 0.00001
        self.M = torch.zeros(1, DIMENSIONS).to(device)
        
        # self.project = embeddings.Sinusoid(SIZE, DIMENSIONS).to(device)
        self.project = embeddings.Projection(SIZE, DIMENSIONS).to(device)

    def encode(self, x):
        sample_hv = self.project(x.T)
        return torchhd.hard_quantize(sample_hv)
        

    def model_update(self, x, y):
        update = self.M + self.lr * (y - (F.linear(x, self.M))) * x
        update = update.mean(0)
        self.M = update

    def forward(self, x) -> Tensor:
        enc = self.encode(x)
        res = F.linear(enc, self.M)
        return res



class ExpRegHD(object):
    def __init__(self, root_path, seq_len, pred_len, dimensionality, training_files, testing_files, device):
        super(ExpRegHD, self).__init__()
        self.root_path = root_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.training_files = training_files
        self.testing_files = testing_files
        self.device = device
        self.HDC = RegHD(self.seq_len, dimensionality, self.device).to(self.device)

    def create_dataset(self, flag):
        return Dataset_ohio(root_path=self.root_path, flag=flag, seq_len=self.seq_len, pred_len=self.pred_len, training_files=self.training_files, testing_files=self.testing_files)
        
    def train(self):
        t, Ts = self.pred_len, self.seq_len 
        train_data = self.create_dataset(flag="train").data
        for i in tqdm(range(Ts, train_data.shape[0] - t, 1)):
            self.run(train_data, i, mode="train")

    def test(self):
        test_data = self.create_dataset(flag="test").data
        preds, trues, maes, rmses = [], [], [], []
        for i in tqdm( range(self.seq_len, test_data.shape[0] - self.pred_len, self.pred_len) ):  
            pred, true = self.run(test_data, i, mode="test")
            preds.append(pred.detach().cpu())
            trues.append(true.detach().cpu())
            mae, rmse = metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
            maes.append(mae)
            rmses.append(rmse)
            
        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        
        # plt.plot(range(preds.shape[0]), preds)
        # plt.plot(range(trues.shape[0]), trues)
        # plt.show()
        # input()

        MAE = cumavg(maes)
        RMSE = cumavg(rmses)

        mae, rmse = MAE[-1], RMSE[-1]
        print("mae: {} / rmse: {}".format(mae, rmse))
        # f = open("results.txt", "a")
        # f.write("TRAIN: [{}]. TEST: [{}]. --- mae: {} / rmse: {}\n".format(self.training_files, self.testing_files, mae, rmse))
        # f.close()


    def run(self, data, idx, mode):

        def inject(x):
            return torch.normal(mean=x, std=torch.full(x.shape, 0.5))
        
        if mode == "train":
            x = torch.Tensor(data[idx - self.seq_len : idx, :]).to(self.device)
            for j in range(self.pred_len):
                y = torch.Tensor(data[idx + j, :]).to(self.device)
                out = torch.full((1,1), self.HDC(x).item()).to(self.device)             
                x = torch.cat((x, out))[1:, :]
                encoded_hv = self.HDC.encode(x)
                self.HDC.model_update(encoded_hv, y)

        elif mode == "test":
            x = torch.Tensor(data[idx - self.seq_len : idx, :]).to(self.device)
            # x = inject(x)
            Y_true = torch.zeros((self.pred_len, data.shape[1]))
            Y_pred = torch.zeros((self.pred_len, data.shape[1]))
            for j in range(self.pred_len):
                y = torch.Tensor(data[idx + j, :]).to(self.device)
                out = torch.full((1,1), self.HDC(x).item()).to(self.device)
                x = torch.cat((x, out))[1:, :] #.detach()
                Y_true[j] = y.detach()
                Y_pred[j] = out.detach()    
            # Update
            x = torch.Tensor(data[idx - self.seq_len : idx, :]).to(self.device)
            for j in range(self.pred_len):
                y = torch.Tensor(data[idx + j, :]).to(self.device)
                out = torch.full((1,1), self.HDC(x).item()).to(self.device)
                x = torch.cat((x, out))[1:, :]
                encoded_hv = self.HDC.encode(x)
                self.HDC.model_update(encoded_hv, y)

            return Y_pred, Y_true
       