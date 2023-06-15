import os
import numpy as np
import torch
import torch.utils.data
import pickle

class BioactivityDataset(torch.utils.data.Dataset):
    def __init__(self, path, test=False):
        super().__init__()
        self.test = test
        if self.test:
            with open(path, 'rb') as f:
                self.x  = pickle.load(f)
        else:
            with open(path, 'rb') as f:
                self.x , self.y = pickle.load(f)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        fp = self.x[i]
        if self.test:
            return fp
        else:
            line = self.y[i]
            return fp, line

class Collater:
    def __call__(self, samples):
        """ Creates a batch out of samples """
        x, y = zip(*samples)
        x = tuple(torch.tensor(np.array(list(zip(x)))))
        y = tuple(torch.tensor(np.array(list(zip(y)))))
        return torch.stack(x), torch.stack(y)
