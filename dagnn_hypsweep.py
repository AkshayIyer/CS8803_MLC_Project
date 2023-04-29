from yaml import load, dump
from yaml import CLoader as Loader, CDumper as Dumper
import itertools
import copy
import subprocess
import numpy as np
import os
import pandas as pd

rcs = [2.0, 4.0, 8.0]
gc_counts = [4, 8, 16, 32]
dim1s = [32, 64, 128]
dim2s = [16, 32]
lrs = [0.001, 0.002]

# https://pyyaml.org/wiki/PyYAMLDocumentation
base_config = load(open("configs/dagnn_config.yml"), Loader=Loader)
# https://docs.python.org/3/library/itertools.html#itertools.product
configs = list(itertools.product(rcs, gc_counts, dim1s, dim2s, lrs))
num_configs = len(configs)
test_errors = np.zeros(num_configs)
val_errors = np.zeros(num_configs)
for i in range(num_configs):
  new_config = copy.deepcopy(base_config)
  new_config['dataset']['n_neighbors'] = 300
  new_config['task']['verbosity'] = 10
  new_config['optim']['max_epochs'] = 200
  new_config['dataset']['cutoff_radius'] = configs[i][0]
  new_config['model']['gc_count'] = configs[i][1]
  new_config['model']['dim1'] = configs[i][2]
  new_config['model']['dim2'] = configs[i][3]
  new_config['optim']['lr'] = configs[i][4]
  f = open('./configs/hyp_swp_config.yml', 'w')
  f.write(dump(new_config, Dumper=Dumper))
  f.close()
  # https://docs.python.org/3/library/subprocess.html
  a = subprocess.run('python scripts/main.py --run_mode=train --config_path=configs/hyp_swp_config.yml', shell=True)
  folder = os.listdir('./results')[0]
  test_df = pd.read_csv('./results/' + folder + '/test_predictions.csv')
  test_errors[i] = np.mean(np.abs(test_df['prediction'] - test_df['target']))
  val_df = pd.read_csv('./results/' + folder + '/val_predictions.csv')
  val_errors[i] = np.mean(np.abs(val_df['prediction'] - val_df['target']))
  _ = subprocess.run('rm -r results/*', shell=True, check=True, text=True)
  _ = subprocess.run('rm configs/hyp_swp_config.yml', shell=True, check=True, text=True)
  print(str(configs[i]) + ' ' + str(test_errors[i]) + ' ' +  str(val_errors[i]))

print(test_errors)
print(val_errors)
