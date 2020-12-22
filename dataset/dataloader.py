import torch
import os
from torchvision.transforms import transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import glob
from PIL import Image
import torch.nn.functional as F
import random
import numpy as np
import itertools
NO_LABEL= -1

class maskDataset(Dataset):
    def __init__(self, data_dir, batch_size, labeled_percents=20):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.train_path = os.path.join(data_dir, "train")
        self.classes = []
        for d in os.listdir(self.train_path):
            if ".txt" not in d:
                self.classes.append(d)

        self.transform = T.Compose([T.Resize((128,128), interpolation=2),
                            T.RandomRotation(45),
                            T.RandomVerticalFlip(),
                            T.RandomGrayscale(),
                            T.RandomSizedCrop((112,112)),
                            T.ToTensor(),
                            T.Normalize(self.mean, self.std)])

        self.paths = []
        self.true_labels = []
        self.true_idxs = []
        self.relabeled = []
        self.labeled_idxs = []
        self.unlabeled_idxs = []
        ck = 0
        last_cls = None
        for i, c in enumerate(self.classes):
            for file in list(glob.glob(os.path.join(self.train_path, os.path.join(c, "*.*")))):
                self.paths.append(file)
                self.true_labels.append(c)
                self.true_idxs.append(i)
                self.relabeled.append(i)
            unlabeled_limit = int((len(self.true_idxs) - ck)*labeled_percents)
            for idx in range(ck,ck+unlabeled_limit):
                self.relabeled[idx] = NO_LABEL
            ck = len(self.relabeled)
        # collect indexes of labels both labeled and unlabeled
        for i, l in enumerate(self.relabeled):
            if l == -1:
                self.unlabeled_idxs.append(i)
            else:
                self.labeled_idxs.append(i)
        # # shuffle labels both labeled and unlabeled
        # random.shuffle(self.labeled_idxs)
        # random.shuffle(self.unlabeled_idxs)
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        X_path = self.paths[idx]
        X = Image.open(X_path).convert("RGB")
        # Y = self.paths[idx][:-1].split(os.path.sep)[-2]
        X = self.transform(X)
        Y = self.relabeled[idx]
        # Y = torch.tensor(Y, dtype=torch.long)
        data = [X, Y]
        return data

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.primary_batch_size = batch_size - secondary_batch_size
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            secondary_batch + primary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    args = [iter(iterable)]*n
    return zip(*args)


            

