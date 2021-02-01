import pathlib
import time
from typing import Dict

import tensorflow as tf
import utils
import numpy as np

from models.ncp_model import SteeringImitationModel, MotorSteeringImitationModel, ConvNCP


def create_log_dir(model_name, base_dir=pathlib.Path('log')):
  outdir = base_dir / f'{model_name}_{time.time()}'
  outdir.mkdir(parents=True, exist_ok=True)
  writer = tf.summary.create_file_writer(str(outdir))
  return outdir, writer

def prepare_data(datadir, motor_neurons, seq_len, validation_size, batch_size):
  data_x, data_y = utils.load_episodes(datadir, motor_neurons)
  print(f"[Info] Loaded {len(data_y)} episodes")

  # prepare data
  data_x, data_y = utils.create_sequences(data_x, data_y, length=seq_len)
  print(f"[Info] Created {len(data_y)} sequences")
  train_x, train_y, val_x, val_y = utils.split_and_shuffle(data_x, data_y, validation_size=validation_size)
  train_x = np.expand_dims(train_x, -1).astype(np.float32)
  val_x = np.expand_dims(val_x, -1).astype(np.float32)
  train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(batch_size)
  val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(batch_size)
  return train_dataset, val_dataset

def get_hparam_value(hparams, param_name):
  return [hparams[h] for h in hparams if h.name == param_name][0]

def create_model(model_name: str, hparams: Dict, motor_neurons: int=1):
  filters, kernels, strides = [], [], []
  base_kernel_size = get_hparam_value(hparams, 'base_kernel_size')
  n_conv_layers = get_hparam_value(hparams, 'n_conv_layers')
  if n_conv_layers == 3:
    filters = [18, 20, 22, 24, 25]
    kernels = [2*base_kernel_size] * 3 + [base_kernel_size] * 2
    strides = [3, 2, 2, 1, 1]
  elif n_conv_layers == 5:
    filters = [18, 20, 22]
    kernels = [10, 10, 10]
    strides = [3, 2, 2]

  if motor_neurons==1:
    model = SteeringImitationModel(name=model_name, conv_filters=filters, conv_kernels=kernels, conv_strides=strides,
                                   motor_neurons=motor_neurons)
  else:
    model = MotorSteeringImitationModel(name=model_name, conv_filters=filters, conv_kernels=kernels, conv_strides=strides,
                                        motor_neurons=motor_neurons)
  return model

def save_model(model: ConvNCP, outdir: pathlib.Path):
  checkpoint_dir = outdir / 'checkpoints'
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  for component in [model._head, model._ncp]:
    component.save(checkpoint_dir / f'{component._name}.pkl')  # store each component separately
  model.save(checkpoint_dir / 'variables.pkl')  # store also the whole model

@tf.function
def train_step(model, x_batch_train, y_batch_train, loss_fn, optimizer):
  with tf.GradientTape() as tape:
    pred = model(x_batch_train)
    loss_value = loss_fn(y_batch_train, pred)
  grads = tape.gradient(loss_value, model.variables)
  optimizer.apply_gradients(zip(grads, model.variables))
  return loss_value

def test_on_track(model, outdir, motor_neurons=1, action_repeat=8, rendering=True):
  video = utils.simulate_episode(model, motor_neurons=motor_neurons, action_repeat=action_repeat, rendering=rendering)
  videodir = outdir / 'videos'
  videodir.mkdir(parents=True, exist_ok=True)
  import imageio
  writer = imageio.get_writer(videodir / 'test.mp4')
  for image in video:
    writer.append_data(image)
  writer.close()

def train_loop(model, train_dataset, val_dataset, epochs, optimizer, loss_fn, writer):
  train_loss = tf.keras.metrics.Mean()
  val_loss = tf.keras.metrics.Mean()
  for epoch in range(epochs):
    init = time.time()
    train_loss.reset_states()
    val_loss.reset_states()

    # Iterate over the batches of the dataset.
    print(f'Epoch {epoch + 1}/{epochs}')
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
      loss_value = model.train_step(x_batch_train, y_batch_train)
      train_loss.update_state(loss_value)
      if step % 25 == 0:
        print(f"\tBatch {step + 1}: train loss: {train_loss.result()}")
    val_loss.update_state(model.evaluate(val_dataset, loss_fn))
    with writer.as_default():
      tf.summary.scalar("train_loss", train_loss.result(), step=epoch)
      tf.summary.scalar("val_loss", val_loss.result(), step=epoch)

    print(f'Epoch {epoch + 1}: train loss: {train_loss.result()}, ' \
          f'validation loss: {val_loss.result()}, time: {time.time() - init}')
  return model
