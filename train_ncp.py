import datetime
import pathlib
import time
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from yamldataclassconfig import YamlDataClassConfig

from training import train, evaluate, utils


def prepare_datasets(datadir, validation_size, motor_neurons, batch_size, seq_len):
    train_dataset, val_dataset = utils.prepare_data(datadir, motor_neurons, seq_len, validation_size, batch_size)
    return train_dataset, val_dataset


def train_once(epochs, train_dataset, val_dataset, hparams, base_log_dir):
    model_name = 'steering_conv_ncp'
    outdir, writer = utils.create_log_dir(model_name, hparams, base_log_dir)
    # define model, optimizer, loss
    model = utils.create_model(model_name, hparams, outdir)
    # training
    print(f'[Info] Initial evaluation: loss: {model.evaluate(val_dataset):.5f}')
    train.train_loop(model, train_dataset, val_dataset, epochs, writer, hparams)
    print(f'[Info] Final evaluation: loss: {model.evaluate(val_dataset):.5f}')
    utils.save_model(model, outdir)
    return model, outdir


def load_hparams_file():
    config = HParamsConfig()
    config.load('hyperparams/ncp.yml')
    STATIC, HPARAMS = {}, []
    for i, (key, values) in enumerate(config.to_dict().items()):
        if i > 0:  # skip first entry
            if type(values) == list:
                HPARAMS.append(hp.HParam(key, hp.Discrete(values)))
            else:
                STATIC[key] = values
    return STATIC, HPARAMS


def tune_hparams(epochs, train_dataset, val_dataset, HPARAMS, logdir):
    with tf.summary.create_file_writer(str(logdir)).as_default():
        hp.hparams_config(
            hparams=HPARAMS,
            metrics=[hp.Metric('train_loss', display_name='TrainMSELoss'),
                     hp.Metric('val_loss', display_name='ValMSELoss')]
        )
    with open(logdir / 'config.txt', 'w') as f:
        content = "\n".join([param.name + ":" + str(param.domain.values) for param in HPARAMS])
        f.write(content)

    session_num = 0
    import itertools
    domains = [param.domain.values for param in HPARAMS]
    for assignment in itertools.product(*domains):
        hparams = {param: val for param, val in zip(HPARAMS, assignment)}
        print(f'--- Starting trial: run-{session_num}')
        print({h.name: hparams[h] for h in hparams})
        model, outdir = train_once(epochs=epochs, train_dataset=train_dataset, val_dataset=val_dataset,
                                   hparams=hparams, base_log_dir=logdir)
        evaluate.test_on_track(model, outdir)
        session_num += 1


@dataclass
class HParamsConfig(YamlDataClassConfig):
    batch_size: int = 32
    seq_len: int = 100
    encoded_dim: int = 32
    motor_neurons: int = 2
    lr: List[float] = field(default_factory=lambda: [1e-3])
    n_conv_layers: List[int] = field(default_factory=lambda: [3])
    base_kernel_size: List[int] = field(default_factory=lambda: [5])
    inter_neurons: List[int] = field(default_factory=lambda: [20])
    command_neurons: List[int] = field(default_factory=lambda: [10])
    sensory_fanout: List[int] = field(default_factory=lambda: [6])
    inter_fanout: List[int] = field(default_factory=lambda: [4])
    recurrent_command_synapses: List[int] = field(default_factory=lambda: [6])
    motor_fanin: List[int] = field(default_factory=lambda: [4])


def main(args):
    datadir = pathlib.Path('data/collect_1612207620.541114/episodes')  # where are located the training data
    if args.mode == 'default':
        # logdir = pathlib.Path('logs')
        # hparams = {
        #    HP_INTER_NEURONS: 24,
        #    HP_COMMAND_NEURONS: 12,
        #    HP_MOTOR_NEURONS: 2,
        #    HP_CONV_LAYERS: 3,
        #    HP_BASE_KERNEL_SZ: 5,
        #    HP_LR: 1e-3,
        #    HP_BATCH_SZ: 32,
        #    HP_SEQ_LEN: 50
        # }
        # model, outdir = train_once(epochs=args.epochs, validation_size=0.15, hparams=hparams,
        #                           datadir=datadir, base_log_dir=logdir)
        # evaluate.test_on_track(model, outdir)
        raise NotImplementedError("not implemented single run")
    elif args.mode == 'hparams':
        datetime_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logdir = pathlib.Path(f'logs/hparams_{datetime_suffix}')
        STATIC, HPARAMS = load_hparams_file()
        train_dataset, val_dataset = prepare_datasets(datadir, validation_size=0.15,
                                                      motor_neurons=STATIC['motor_neurons'],
                                                      batch_size=STATIC['batch_size'], seq_len=STATIC['seq_len'])
        tune_hparams(epochs=args.epochs, train_dataset=train_dataset, val_dataset=val_dataset, HPARAMS=HPARAMS,
                     logdir=logdir)
    else:
        raise NotImplementedError(f'mode {args.mode} not implemented')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--mode', choices=['default', 'hparams'], required=True)
    parser.add_argument('--epochs', type=int, required=True, default=50)
    args = parser.parse_args()
    main(args)
