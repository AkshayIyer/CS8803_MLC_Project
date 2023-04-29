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

        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")
        currday, currtime = now.split('_')

        if self.config['save_files']:
            if not os.path.exists('saved_models/' + now + '/'):
                os.makedirs('saved_models/' + now + '/')
            self.save_path = 'saved_models/' + now + '/'

            with open(self.save_path + 'config.json', 'w') as fp:
                json.dump(self.config, fp)

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
        data = data.to(self.device)
        pred_y, pred_noise = model(data.x, data.pos, data.batch, data)

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

        list_train_loss = []
        list_train_y_loss = []
        list_train_noise_loss = []
        list_valid_loss = []
        list_valid_y_loss = []
        list_valid_noise_loss = []

        for epoch_counter in range(self.config['epochs']):
            train_loss_y = 0.
            train_loss_noise = 0.
            train_loss = 0.

            for bn, data in enumerate(train_loader):                
                adjust_learning_rate(optimizer, epoch_counter + bn / len(train_loader), self.config)

                loss_y, loss_noise = self.loss_fn(model, data)
                loss = loss_y + loss_noise

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_y += loss_y.item()
                train_loss_noise += loss_noise.item()
                train_loss += loss.item()

                # if n_iter % self.config['log_every_n_steps'] == 0:
                #     log_str = 'Epoch: {:04d}, Train loss: {:.5f}, Loss y: {:.5f}, Loss noise: {:.5f}'.format(
                #         epoch_counter, loss.item(), loss_y.item(), loss_noise.item()
                #     )
                #     print(log_str)
                #     torch.cuda.empty_cache()
                #     gc.collect() # free memory

                n_iter += 1
            
            train_loss_y /= (bn+1)
            train_loss_noise /= (bn+1)
            train_loss /= (bn+1)

            gc.collect() # free memory
            torch.cuda.empty_cache()

            # validate the model 
            valid_loss, valid_loss_y, valid_loss_noise = self._validate(model, valid_loader)
            log_str = 'Epoch: {:04d}, Train loss: {:.5f}, Loss y: {:.5f}, Loss noise: {:.3f} || Valid loss: {:.5f}, Loss y: {:.5f}, Loss noise: {:.3f}'.format(
                        epoch_counter, train_loss, train_loss_y, train_loss_noise, valid_loss, valid_loss_y, valid_loss_noise
                    )
            print(log_str)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if self.config['save_files']:
                    torch.save(model.state_dict(), os.path.join(self.save_path, 'model.pth'))

            valid_n_iter += 1

            # append losses to lists
            list_train_loss.append(train_loss)
            list_train_y_loss.append(train_loss_y)
            list_train_noise_loss.append(train_loss_noise)
            list_valid_loss.append(valid_loss)
            list_valid_y_loss.append(valid_loss_y)
            list_valid_noise_loss.append(valid_loss_noise)

        # save losses to csv
        if self.config['save_files']:
            df = pd.DataFrame({
                'train_loss': list_train_loss,
                'train_y_loss': list_train_y_loss,
                'train_noise_loss': list_train_noise_loss,
                'valid_loss': list_valid_loss,
                'valid_y_loss': list_valid_y_loss,
                'valid_noise_loss': list_valid_noise_loss,
            })
            df.to_csv(os.path.join(self.save_path, 'losses.csv'), index=False)

        print("Best val loss: {:.5f}".format(best_valid_loss))
        print("=========================================")

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

    # num_layers = [4, 6, 8, 10, 12]
    # num_heads = [2, 4, 6, 8, 10]
    # cutoff_upper = [4, 6, 8, 10, 12, 16]

    config['save_files'] = False
    # config['model']['num_layers'] = n_layer
    # config['model']['cutoff_upper'] = r

    print(config)

    input()

    gc.collect()
    torch.cuda.empty_cache()

    trainer = Trainer(config)
    trainer.train()

    # for n_layer in num_layers:
    #     for r in cutoff_upper:
    #         config['model']['num_layers'] = n_layer
    #         config['model']['cutoff_upper'] = r

    #         print(config)

    #         gc.collect()
    #         torch.cuda.empty_cache()

    #         trainer = Trainer(config)
    #         trainer.train()