from typing import List

import numpy as np
import pathlib

def load_episodes(path: pathlib.Path):
  x, y = [], []
  for file in path.glob('*.npy'):
    observations = []
    actions = []
    data = np.load(file, allow_pickle=True)
    for transition in data:
      observations.append(transition['observation']['lidar'])
      actions.append(np.array([transition['action']['steering']]))
    x.append(np.array(observations))
    y.append(np.array(actions))
  return x, y

def create_sequences(data_x, data_y, length=50):
  sequences_x, sequences_y = [], []
  for x, y in zip(data_x, data_y):
    tot_len = len(y)
    for start in range(0, tot_len-length, length//10):
      seq_x = x[start:start+length, :]
      seq_y = y[start:start + length, :]
      sequences_x.append(seq_x)
      sequences_y.append(seq_y)
  return sequences_x, sequences_y

def split_and_shuffle(x: List, y: List, validation_size: float=0.10):
  n = len(y)
  x = np.array(x)
  y = np.array(y)
  ids = np.arange(0, n)
  np.random.shuffle(ids)
  validation_x, validation_y = x[ids[:int(validation_size*n)]], y[ids[:int(validation_size*n)]]
  train_x, train_y = x[ids[int(validation_size * n):]], y[ids[int(validation_size * n):]]
  return train_x, train_y, validation_x, validation_y




