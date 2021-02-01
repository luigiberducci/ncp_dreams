import pathlib
import time
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from models.ncp_model import ConvNCP
import utils
import training

# training params
HP_LR = hp.HParam('lr', hp.Discrete([1e-3, 1e-4, 1e-5]))
# conv head params
HP_CONV_LAYERS = hp.HParam('n_conv_layers', hp.Discrete([3, 5]))
HP_BASE_KERNEL_SZ = hp.HParam('base_kernel_size', hp.Discrete([3, 5, 7, 10]))
# ncp params
HP_INTER_NEURONS = hp.HParam('inter_neurons', hp.Discrete([12]))
HP_COMMAND_NEURONS = hp.HParam('command_neurons', hp.Discrete([19]))
HP_MOTOR_NEURONS = hp.HParam('motor_neurons', hp.Discrete([1, 2]))
HP_SENSORY_FANOUT = hp.HParam('sensory_fanout', hp.Discrete([6]))
HP_INTER_FANOUT = hp.HParam('inter_fanout', hp.Discrete([6]))
HP_RECURRENT_COMMAND_SYN = hp.HParam('recurrent_cmd_synapses', hp.Discrete([6]))
HP_MOTOR_FANIN = hp.HParam('motor_fanin', hp.Discrete([4]))
HPARAMS = [HP_LR, HP_CONV_LAYERS, HP_BASE_KERNEL_SZ,
           HP_INTER_NEURONS, HP_COMMAND_NEURONS, HP_MOTOR_NEURONS,
           HP_SENSORY_FANOUT, HP_INTER_FANOUT, HP_RECURRENT_COMMAND_SYN, HP_MOTOR_FANIN]

def train_once(seq_len, epochs, batch_size, validation_size, hparams, datadir, base_log_dir):
    # initialization
    motor_neurons = hparams[HP_MOTOR_NEURONS]
    model_name = 'steering_conv_ncp' if motor_neurons == 1 else 'motor_steering_conv_ncp'
    outdir, writer = training.create_log_dir(model_name, base_log_dir)
    train_dataset, val_dataset = training.prepare_data(datadir, motor_neurons, seq_len, validation_size, batch_size)
    # define model, optimizer, loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=hparams[HP_LR])
    loss_fn = tf.losses.MeanSquaredError()
    model = training.create_model(model_name, hparams, motor_neurons)
    # training
    print(f'[Info] Initial evaluation: loss: {model.evaluate(val_dataset, loss_fn)}')
    training.train_loop(model, train_dataset, val_dataset, epochs, optimizer, loss_fn, writer)
    print(f'[Info] Final evaluation: loss: {model.evaluate(val_dataset, loss_fn)}')
    return model, outdir

def tune_hparams(datadir, logdir):
    METRIC_MSE = "mse"
    with tf.summary.create_file_writer(str(logdir)).as_default():
        hp.hparams_config(
            hparams=HPARAMS,
            metrics=[hp.Metric(METRIC_MSE, display_name='MeanSquaredError')],
        )

    session_num = 0
    for motor_neurons in HP_MOTOR_NEURONS.domain.values:
        for conv_layers in HP_CONV_LAYERS.domain.values:
            for base_conv_kernel in HP_BASE_KERNEL_SZ.domain.values:
                for lr in HP_LR.domain.values:
                    hparams = {
                        HP_MOTOR_NEURONS: motor_neurons,
                        HP_CONV_LAYERS: conv_layers,
                        HP_BASE_KERNEL_SZ: base_conv_kernel,
                        HP_LR: lr
                    }
                    print(f'--- Starting trial: run-{session_num}')
                    print({h.name: hparams[h] for h in hparams})
                    train_once(seq_len=50, epochs=1, batch_size=32, validation_size=0.15, hparams=hparams,
                               datadir=datadir, base_log_dir=logdir)
                    session_num += 1


def main(args):
    datadir = pathlib.Path('data/collect_1612207620.541114/episodes')
    if args.mode == 'default':
        logdir = pathlib.Path('logs')
        hparams = {
            HP_MOTOR_NEURONS: args.motors,
            HP_CONV_LAYERS: 5,
            HP_BASE_KERNEL_SZ: 5,
            HP_LR: 1e-3
        }
        model, outdir = train_once(seq_len=50, epochs=args.epochs, batch_size=32, validation_size=0.15, hparams=hparams,
                                   datadir=datadir, base_log_dir=logdir)
        training.test_on_track(model, outdir, motor_neurons=args.motors, rendering=True)
    elif args.mode == 'hparams':
        logdir = pathlib.Path('logs/hparams')
        tune_hparams(datadir, logdir)
    else:
      raise NotImplementedError(f'mode {args.mode} not implemented')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--mode', choices=['default', 'hparams'])
    parser.add_argument('--epochs', type=int, required=False, default=50)
    parser.add_argument('--motors', type=int, required=False, default=1)
    args = parser.parse_args()
    main(args)
