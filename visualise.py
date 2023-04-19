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

from models.cgcnn import CGCNN
from models.torchmdnet import TorchMD_ET
from dataset_utils.datasets import DataWrapper

def get_predictions(save_path, config):
    model_name = config['model']['name']
    if model_name == "CGCNN":
        model = CGCNN(**config["model"])
    elif model_name == "TorchMD_ET":
        model = TorchMD_ET(**config["model"])
    model.load_state_dict(torch.load(save_path + 'model.pth'))
    model.eval()

    test_path = './data/test_data.json'
    data_wrapper = DataWrapper(batch_size=128, num_workers=0, valid_size=0.1, std=0.1)
    test_dataloader = data_wrapper.get_dataloader(test_path)

    predictions = []
    targets = []

    for bn, data in enumerate(test_dataloader):
        if model_name == "CGCNN":
            with torch.no_grad():
                pred_y = model(data)
        elif model_name == "TorchMD_ET":
            with torch.no_grad():
                pred_y = model(data.x, data.pos, data.batch, data)

        predictions.append(pred_y)
        targets.append(data.y)

    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    return predictions.detach().cpu().numpy(), targets.detach().cpu().numpy()

# save_path = 'saved_models/2023-04-18_23-05-57/'
save_path = 'saved_models/2023-04-19_00-04-11/'
config = yaml.load(open("configs/cgcnn_config.yml", "r"), Loader=yaml.FullLoader)

predictions, targets = get_predictions(save_path, config)

# for i in range(len(predictions)):
#     print(predictions[i], targets[i])

# calculate MAE between predictions and targets
mae = np.mean(np.abs(predictions - targets))
print("MAE:", mae)