import os
import gc
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

class PrepareData(Dataset):
    def __init__(
        self, 
        species, 
        positions, 
        list_y,
        list_natoms,
        list_cell,
        std
    ):
        self.species = species
        self.positions = positions
        self.list_y = list_y
        self.list_natoms = list_natoms
        self.list_cell = list_cell
        self.std = std
    
    def __getitem__(self, index):
        ori_pos = self.positions[index]
        atoms = self.species[index]
        targets = self.list_y[index]
        natoms = self.list_natoms[index]
        cells = self.list_cell[index]

        # add noise
        noise = np.random.normal(0, self.std, ori_pos.shape)
        pos = ori_pos + noise

        x = torch.tensor(atoms, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        noise = torch.tensor(noise, dtype=torch.float)
        y = torch.tensor(targets, dtype=torch.float)
        cells = torch.tensor(cells, dtype=torch.float)
        cells = cells.view(1, 3, 3)
        
        data = Data(x=x, pos=pos, y=y, noise=noise, cell=cells, natoms=natoms)
        return data

    def __len__(self):
        return len(self.positions)
    
class DataWrapper(object):
    def __init__(self, batch_size, num_workers, std, paths=None, seed=1234, **kwargs):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.std = std
        self.paths = paths
        self.seed = seed

    def get_dataloader(self, path):
        f = open(path, 'r')
        data = json.load(f)
        f.close()

        list_atomic_numbers = []
        list_positions = []
        list_y = []
        list_natoms = []
        list_cell = []

        for i, d in enumerate(data):
            list_atomic_numbers.append(d['atomic_numbers'])
            list_positions.append(np.array(d['positions']))
            list_y.append(d['y'])
            list_natoms.append(len(d['atomic_numbers']))
            list_cell.append(d['cell'])

        dataset = PrepareData(
            list_atomic_numbers, list_positions, list_y, list_natoms, list_cell, self.std
        )
        dataloader = PyGDataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, drop_last=True, pin_memory=True, persistent_workers=(self.num_workers > 0)
        )

        gc.collect()
        return dataloader
        
    def get_all_loaders(self):
        train_dataloader = self.get_dataloader(self.paths['train'])
        valid_dataloader = self.get_dataloader(self.paths['val'])
        test_dataloader = self.get_dataloader(self.paths['test'])

        return train_dataloader, valid_dataloader, test_dataloader