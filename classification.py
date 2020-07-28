import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class PosesDataset(Dataset):
    def __init__(self, csv_file):
        self.poses_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.poses_frame)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        poses = self.poses_frame.iloc[idx, 1:]
        poses = np.array([poses])

        sample = {'time': self.poses_frame.iloc[idx, 0], 'poses': poses}
        
        return sample


pose_data = PosesDataset(csv_file='data/1_Teleport_0_10_poses.csv')

print(len(pose_data))


