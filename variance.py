import os
import gc
import csv
import shutil
import yaml
import math
import json
import numpy as np
import datetime
import pandas as pd

import torch
from torch.optim import AdamW
import torch.nn.functional as F

from models.cgcnn_var import CGCNN
from dataset_utils.datasets import DataWrapper

def get_representations(save_path, config):
    model = CGCNN(**config["model"])
    model.load_state_dict(torch.load(save_path + 'model.pth'))
    model.eval()

    test_path = './data/test_data.json'
    data_wrapper = DataWrapper(batch_size=1, num_workers=0, valid_size=0.1, std=0.1)
    test_dataloader = data_wrapper.get_dataloader(test_path)

    all_reps = []

    for bn, data in enumerate(test_dataloader):
        with torch.no_grad():
            reps = model(data)
            all_reps.append(reps)
        break
    
    all_reps = torch.cat(all_reps, dim=0)
    return all_reps

def get_var(x, reduction="sum"):
    v = torch.var(x, dim=0)
    if reduction == "sum":
        return torch.sum(v)
    elif reduction == "mean":
        return torch.mean(v)


paths = ['2023-04-19_23-15-39',
'2023-04-19_23-21-04',
'2023-04-19_23-26-51',
'2023-04-19_23-33-20',
'2023-04-19_23-40-06',
'2023-04-19_23-47-05',
'2023-04-19_23-55-27',]

paths = ['2023-04-20_01-13-58',
'2023-04-20_01-35-39',
'2023-04-20_01-58-31',
'2023-04-20_02-23-14',
'2023-04-20_02-54-52']

num_gc_layers = [2, 4, 6, 8, 10, 16, 32]
num_gc_layers = [4, 6, 8, 16, 32]

for p, n in zip(paths, num_gc_layers):
    save_path = 'saved_models/' + p + '/'
    config = yaml.load(open("configs/cgcnn_config.yml", "r"), Loader=yaml.FullLoader)
    config['model']['num_gc_layers'] = n
    config['dataset']['batch_size'] = 1
    reps = get_representations(save_path, config)
    print(get_var(reps, reduction="sum"))
