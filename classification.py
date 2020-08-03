import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# each dataset contains the data for a single person, so we'll need to concat datasets
# that means we should create a new column containing the user identifier
class PosesDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.poses_frame = pd.read_csv(csv_file)
        self.transform = transform

        # parse user from csv_file name
        user = csv_file.split('_')[3]
        self.poses_frame['user'] = [user] * len(self.poses_frame)

    def __len__(self):
        return len(self.poses_frame)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        user = self.poses_frame.iloc[idx]['user']
        poses = self.poses_frame.iloc[idx, 1:-1]
        poses = np.array([poses])

        sample = {'user': user, 'poses': poses}
        
        return sample

class ThrowsDataset(Dataset):
    def __init__(self, poses_file, events_file, transform=None):
        self.poses_frame = pd.read_csv(poses_file)
        self.events_frame = pd.read_csv(events_file)
        self.transform = transform

        # parse user from csv_file name
        user = poses_file.split('_')[3]
        self.poses_frame['user'] = [user] * len(self.poses_frame)
        

        grabs = self.events_frame[self.events_frame['event'] == 'GRABBED_OBJECT']
        throws = self.events_frame[self.events_frame['event'] == 'TARGET_THROW']
        # for the time between a grab and a release, get the poses
        starttimes = []
        for _, grab in grabs.iterrows():
            starttimes.append(grab['time'])

        endtimes = []
        for _, grab in throws.iterrows():
            endtimes.append(grab['time'])

        times = zip(starttimes, endtimes)

        self.throws_poses = pd.DataFrame()
        for (start, end) in times:
            throw = self.poses_frame[(self.poses_frame['time'] > start) & (self.poses_frame['time'] < end)]
            self.throws_poses = self.throws_poses.append(throw, ignore_index = True)

    def __len__(self):
        return len(self.throws_poses)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        user = self.poses_frame.iloc[idx]['user']
        poses = self.throws_poses.iloc[idx, 1:-1]
        poses = np.array([poses])

        sample = {'user': user, 'throw pose': poses}



class ToTensor(object):
    """ convert ndarrays to Tensors """

    def __call__(self, sample):
        user, poses = sample['user'], sample['poses']

        return {'user': torch.from_numpy(user), 'poses': torch.from_numpy(poses)}
        


pose_data = PosesDataset(csv_file='data/1_Teleport_0_10_poses.csv')
print(pose_data[0])

events_file = 'data/1_HandWalking_0_133_1596455970log/1_HandWalking_0_133_events.csv'
poses_file = 'data/1_HandWalking_0_133_1596455970log/1_HandWalking_0_133_poses.csv'

throws_data = ThrowsDataset(poses_file=poses_file, events_file=events_file)

