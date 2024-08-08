import os
import warnings
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

def remove_nan_strat_end(df, attr):
    index_start = 0
    index_end = 0
    if attr in list(df):
        for i in range(len(df) - 1):
            if np.isnan(df[attr][i]) == False:
                index_start = i
                break
        for j in range(len(df) - 1, 0, -1):
            if np.isnan(df[attr][j]) == False:
                index_end = j + 1
                break
        df = df[index_start:index_end]
        df = df.reset_index(drop=True)
        return df
    else:
        print('Unable to remove: No Attibutes Found\n')
        return 0

def filling_CGM(testpath):
    filepath = "../dataset/processedcsv/ohio540.csv"
    a_test = pd.read_csv(filepath, usecols=['CGM'])
    a_test = remove_nan_strat_end(a_test, 'CGM')  
    AA = a_test['CGM'].fillna(1)
    return AA



class Dataset_ohio(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        training_files=None,
        testing_files=None
    ):
    
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.root_path = root_path
        self.flag = flag
        self.training_files = training_files
        self.testing_files = testing_files
        self.__read_data__()


    def __read_data__(self):
        df_raw = []
        if(self.flag == "train"):
            for file in self.training_files:
                full_path = os.path.join(self.root_path, file)
                df_raw.append( pd.read_csv(full_path) )
        else:
            for file in self.testing_files:
                full_path = os.path.join(self.root_path, file)
                df_raw.append( pd.read_csv(full_path) )

        df_raw = pd.concat(df_raw)
    
        df_data = df_raw[["Time", 'CGM']] 
        # df_data['CGM'] = filling_CGM(df_data)
        df_data = df_data['CGM'].to_frame()

        self.data = df_data.values
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
