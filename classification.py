import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class PosesDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.poses_frame = pd.read_csv(csv_file)
        self.transform = transform

        # parse user from csv_file name
        self.user = csv_file.split('_')[3]

    def __len__(self):
        return len(self.poses_frame)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        poses = self.poses_frame.iloc[idx, 1:]
        poses = np.array([poses])

        sample = {'user': self.user, 'poses': poses}
        
        return sample

class ToTensor(object):
    """ convert ndarrays to Tensors """

    def __call__(self, sample):
        user, poses = sample['user'], sample['poses']

        return {'user': torch.from_numpy(user), 'poses': torch.from_numpy(poses)}
        


pose_data = PosesDataset(csv_file='data/1_Teleport_0_10_poses.csv')

print(len(pose_data))


