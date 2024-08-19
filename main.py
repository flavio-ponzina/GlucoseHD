import argparse
import os
import numpy as np
import torch
from reghd import ExpRegHD

parser = argparse.ArgumentParser(description="[Informer] Long Sequences Forecasting")
parser.add_argument("--root_path", type=str, default="./dataset/processedcsv/", help="root path of the data file")
parser.add_argument("--seq_len", type=int, default=30, help="input sequence length of Informer encoder")
parser.add_argument("--pred_len", type=int, default=6, help="prediction sequence length/forecast window")
parser.add_argument("--dimensionality", type=int, default=30000, help="dimension of the hypervectors")
args = parser.parse_args()

training_files = [
                      "ohio540_Training.csv",
                      "ohio544_Training.csv",
                      "ohio552_Training.csv",
                      "ohio559_Training.csv",
                      "ohio563_Training.csv",
                      "ohio567_Training.csv",
                      "ohio570_Training.csv",
                      "ohio575_Training.csv",
                      "ohio584_Training.csv",
                      "ohio588_Training.csv",  
                      "ohio591_Training.csv",
                      "ohio596_Training.csv"
    ]

testing_files =  [
                      "ohio540_Testing.csv",
                      # "ohio544_Testing.csv",
                      # "ohio552_Testing.csv",
                      # "ohio559_Testing.csv",
                      # "ohio563_Testing.csv",
                      # "ohio567_Testing.csv",
                      # "ohio570_Testing.csv",
                      # "ohio575_Testing.csv",
                      # "ohio584_Testing.csv",
                      # "ohio588_Testing.csv",  
                      # "ohio591_Testing.csv",
                      # "ohio596_Testing.csv"
    ]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

exp = ExpRegHD(args.root_path, args.seq_len, args.pred_len, args.dimensionality, training_files, testing_files, device)
exp.train()
exp.test()     