import pathlib
import time
from typing import List, Dict
import tensorflow as tf

import numpy as np
import yaml

from models.ncp_model import ConvNCP


def load_episodes(path: pathlib.Path, motor_neurons: int):
    x, y = [], []
    for file in path.glob('*.npy'):
        observations = []
        actions = []
        data = np.load(str(file), allow_pickle=True)
        for transition in data:
            observations.append(transition['observation']['lidar'])
            if motor_neurons == 1:
                action = [transition['action']['steering']]
            else:
                action = list(transition['action'].values())
            actions.append(np.array(action))
        x.append(np.array(observations))
        y.append(np.array(actions))
    return x, y


def create_sequences(data_x: List, data_y: List, length: int = 50):
    sequences_x, sequences_y = [], []
    for x, y in zip(data_x, data_y):
        tot_len = len(y)
        for start in range(0, tot_len - length, 5):
            seq_x = x[start:start + length, :]
            seq_y = y[start:start + length, :]
            sequences_x.append(seq_x)
            sequences_y.append(seq_y)
    return sequences_x, sequences_y


def split_and_shuffle(x: List, y: List, validation_size: float = 0.10):
    n = len(y)
    x = np.array(x)
    y = np.array(y)
    ids = np.arange(0, n)
    np.random.shuffle(ids)
    validation_x, validation_y = x[ids[:int(validation_size * n)]], y[ids[:int(validation_size * n)]]
    train_x, train_y = x[ids[int(validation_size * n):]], y[ids[int(validation_size * n):]]
    return train_x, train_y, validation_x, validation_y


def create_log_dir(model_name: str, hparams: Dict, base_dir: pathlib.Path = pathlib.Path('log')):
    outdir = base_dir / f'{model_name}_{time.time()}'
    cp_dir = outdir / 'checkpoints'
    cp_dir.mkdir(parents=True, exist_ok=True)
    writer = tf.summary.create_file_writer(str(outdir))
    with open(outdir / 'hparameters.txt', 'w') as f:
        yaml.dump({p.name: val for p, val in hparams.items()}, f)
    return outdir, writer


def prepare_data(datadir: pathlib.Path, motor_neurons: int, seq_len: int, validation_size: float, batch_size: int):
    data_x, data_y = load_episodes(datadir, motor_neurons)
    print(f"[Info] Loaded {len(data_y)} episodes")

    # prepare data
    data_x, data_y = create_sequences(data_x, data_y, length=seq_len)
    print(f"[Info] Created {len(data_y)} sequences")
    train_x, train_y, val_x, val_y = split_and_shuffle(data_x, data_y, validation_size=validation_size)
    # normalize obervations
    train_x = np.clip(train_x, 0, 15.0) / 15.0 - 0.5  # normalize observation in +-0.5
    val_x = np.clip(val_x, 0, 15.0) / 15.0 - 0.5  # normalize observation in +-0.5
    train_x = np.expand_dims(train_x, -1).astype(np.float32)
    val_x = np.expand_dims(val_x, -1).astype(np.float32)
    # tf dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(batch_size)
    return train_dataset, val_dataset


def get_hparam_value(hparams: Dict, param_name: str):
    for h in hparams:
        if h.name == param_name:
            return hparams[h]
    raise Exception(f'parameter {param_name} not found')


def create_model(model_name: str, hparams: Dict, logdir: pathlib.Path):
    filters, kernels, strides = [], [], []
    base_kernel_size = get_hparam_value(hparams, 'base_kernel_size')
    n_conv_layers = get_hparam_value(hparams, 'n_conv_layers')
    inter_neurons = get_hparam_value(hparams, 'inter_neurons')
    command_neurons = get_hparam_value(hparams, 'command_neurons')
    motor_neurons = 2
    cmd_synapses = get_hparam_value(hparams, 'recurrent_command_synapses')
    if n_conv_layers == 5:
        filters = [18, 20, 22, 24, 25]
        kernels = [2 * base_kernel_size] * 3 + [base_kernel_size] * 2
        strides = [3, 2, 2, 1, 1]
    elif n_conv_layers == 3:
        filters = [18, 20, 22]
        kernels = [10, 10, 10]
        strides = [3, 2, 2]

    model = ConvNCP(name=model_name, logdir=logdir, conv_filters=filters, conv_kernels=kernels, encoded_dim=32,
                    conv_strides=strides, inter_neurons=inter_neurons, command_neurons=command_neurons,
                    motor_neurons=motor_neurons, recurrent_command_synapses=cmd_synapses)
    return model


def save_model(model: ConvNCP, outdir: pathlib.Path):
    checkpoint_dir = outdir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for component in [model._head, model._ncp]:
        component.save(checkpoint_dir / f'{component._name}.pkl')  # store each component separately
    model.save(checkpoint_dir / 'variables.pkl')  # store also the whole model
