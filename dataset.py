import torch
from torch.utils.data import Dataset

class transition_dataset(Dataset):
    def __init__(self, transitions):
        self.data = []        
        for transition in transitions:
            s, a, s_prime = transition
            self.data.append([s, a, s_prime]) 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        s, a, s_prime = self.data[idx]
        return s, a, s_prime

class imitation_dataset(Dataset):
    def __init__(self, trajs):
        self.data = []
        for traj in trajs:
            for sample in traj:
                s, a = sample
                self.data.append([s, a[0]])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        s, a = self.data[idx]        
        return s, a