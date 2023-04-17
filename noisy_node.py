import os
import gc
import csv
import shutil
import yaml
import math
import numpy as np
from datetime import datetime

import torch
from torch.optim import AdamW
import torch.nn.functional as F

from models.torchmdnet import TorchMD_ET
from dataset_utils.datasets import DataWrapper

def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config['warmup_epochs']:
        lr = (config['lr'] - config['min_lr']) * epoch / config['warmup_epochs'] + config['min_lr']
    elif epoch < config['warmup_epochs'] + config['patience_epochs']:
        lr = config['lr']
    else:
        prev_epochs = config['warmup_epochs'] + config['patience_epochs']
        lr = config['min_lr'] + (config['lr'] - config['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - prev_epochs) / (config['epochs'] - prev_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.dataset = DataWrapper(**self.config['dataset'])

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    @staticmethod
    def _save_config_file(ckpt_dir):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
            shutil.copy('./config_pretrain.yaml', os.path.join(ckpt_dir, 'config_pretrain.yaml'))

    def loss_fn(self, model, data):
        pred_y, pred_noise = model(data.x, data.pos, data.batch)

        loss_y = F.mse_loss(pred_y, data.y)
        loss_noise = F.mse_loss(pred_noise, data.noise, reduction='sum')

        return loss_y, loss_noise

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_all_loaders()
        model = TorchMD_ET(**self.config["model"])

        model = model.to(self.device)
        model.train()

        if type(self.config['lr']) == str: self.config['lr'] = eval(self.config['lr']) 
        if type(self.config['min_lr']) == str: self.config['min_lr'] = eval(self.config['min_lr'])
        if type(self.config['weight_decay']) == str: self.config['weight_decay'] = eval(self.config['weight_decay']) 
        optimizer = AdamW(
            model.parameters(), self.config['lr'],
            weight_decay=self.config['weight_decay'],
        )

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):                
                adjust_learning_rate(optimizer, epoch_counter + bn / len(train_loader), self.config)

                loss_y, loss_noise = self.loss_fn(model, data)
                loss = loss_y + loss_noise

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if n_iter % self.config['log_every_n_steps'] == 0:
                    print(epoch_counter, bn, 'loss', loss.item(), 'loss_y', loss_y.item(), 'loss_noise', loss_noise.item())
                    torch.cuda.empty_cache()
                    gc.collect() # free memory

                n_iter += 1

            gc.collect() # free memory
            torch.cuda.empty_cache()

            # validate the model 
            valid_loss, valid_loss_y, valid_loss_noise = self._validate(model, valid_loader)
            print('Validation', epoch_counter, 'valid loss', valid_loss, 'valid loss_y', valid_loss_y, 'valid loss_noise', valid_loss_noise)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

            valid_n_iter += 1

        return model

    def _validate(self, model, valid_loader):
        valid_loss_y = 0
        valid_loss_noise = 0
        valid_loss = 0
        model.eval()

        for bn, data in enumerate(valid_loader):                
            loss_y, loss_noise = self.loss_fn(model, data)
            loss = loss_y + loss_noise

            valid_loss_y += loss_y.item()
            valid_loss_noise += loss_noise.item()
            valid_loss += loss.item()
            torch.cuda.empty_cache()
        
        gc.collect() # free memory

        model.train()
        return valid_loss / (bn+1), valid_loss_y / (bn+1), valid_loss_noise / (bn+1)


if __name__ == "__main__":
    config = yaml.load(open("configs/nn_config.yml", "r"), Loader=yaml.FullLoader)
    print(config)

    trainer = Trainer(config)
    trainer.train()