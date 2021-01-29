import pathlib
import time

import numpy as np
import os
from tensorflow import keras
import kerasncp as kncp
import matplotlib.pyplot as plt
import seaborn as sns

import wrappers
from models.ncp_model import NCP
import tensorflow as tf
import utils
import tensorflow.keras.layers as kl

# params
seq_len = 50      # train ncp model on sequences of fixed length
epochs = 50
batch_size = 32

# load data
datadir = pathlib.Path('data/collect_1611939067.9052866/episodes')
data_x, data_y = utils.load_episodes(datadir)
print(f"[Info] Loaded {len(data_y)} episodes")

# prepare data
data_x, data_y = utils.create_sequences(data_x, data_y, length=seq_len)
print(f"[Info] Created {len(data_y)} sequences")
train_x, train_y, val_x, val_y = utils.split_and_shuffle(data_x, data_y, validation_size=0.15)
train_x = np.expand_dims(train_x, -1)
val_x = np.expand_dims(val_x, -1)

# define model
ncp_arch = kncp.wirings.NCP(
            inter_neurons=12,       # Number of inter neurons
            command_neurons=19,     # Number of command neurons
            motor_neurons=1,        # Number of motor neurons
            sensory_fanout=6,       # How many outgoing synapses has each sensory neuron
            inter_fanout=4,         # How many outgoing synapses has each inter neuron
            recurrent_command_synapses=6,  # Now many recurrent synapses are in the command neuron layer
            motor_fanin=4,          # How many incomming syanpses has each motor neuron
        )
ncp_cell = kncp.LTCCell(ncp_arch)
model = tf.keras.models.Sequential(
  [
        keras.layers.InputLayer(input_shape=(None, 1080, 1)),
        keras.layers.TimeDistributed(
            keras.layers.Conv1D(18, 10, strides=3, activation="relu")
        ),
        keras.layers.TimeDistributed(
            keras.layers.Conv1D(20, 10, strides=2, activation="relu")
        ),
        keras.layers.TimeDistributed(keras.layers.MaxPool1D()),
        keras.layers.TimeDistributed(
            keras.layers.Conv1D(22, 10, strides=2, activation="relu")
        ),
        keras.layers.TimeDistributed(keras.layers.MaxPool1D()),
        keras.layers.TimeDistributed(
            keras.layers.Conv1D(24, 5, activation="relu")
        ),
        keras.layers.TimeDistributed(keras.layers.Flatten()),
        keras.layers.TimeDistributed(keras.layers.Dense(32, activation="relu")),
        keras.layers.RNN(ncp_cell, return_sequences=True),
    ])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

model.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
model.summary()

# plot prediction from untrained model
fig, axis = plt.subplots(1, 3)
for i, ax in enumerate(axis):
  pred = model(np.expand_dims(data_x[i], 0))[0]
  ax.plot(range(len(data_y[i])), data_y[i], label='true')
  ax.plot(range(len(data_y[i])), pred, label='untrained')
print(f'[Info] Preliminary evaluation: loss: {model.evaluate(val_x, val_y)}')

# train model
hist = model.fit(train_x, train_y, epochs=epochs, verbose=1, batch_size=batch_size, validation_data=(val_x, val_y))
model.save(f'checkpoints/checkpoint_epochs{epochs}_batch{batch_size}_{time.time()}')

# plot predictions trained model
for i, ax in enumerate(axis):
  pred = model(np.expand_dims(data_x[i], 0))[0]
  ax.plot(range(len(data_y[i])), pred, label='trained')
  ax.legend()
print(f'[Info] Final evaluation: loss: {model.evaluate(val_x, val_y)}')
fig.savefig("samples.png")
plt.show()

# plot loss
sns.set()
plt.figure(figsize=(6, 4))
plt.plot(hist.history["loss"], label="Training loss")
plt.legend(loc="upper right")
plt.xlabel("Training steps")
plt.show()

# test on track
import racecar_gym
import gym


action_repeat = 8
env = gym.make('SingleAgentTreitlstrasse_v2_Gui-v0')
env = wrappers.ActionRepeat(env, action_repeat)

# set variables
import time
init = time.time()
done = False
obs = env.reset(mode='grid')
state = None
t = 0
returns = 0.0
video = []
# simulate
while not done:
  action = {'motor': 0.01, 'steering': 0.00}
  x = np.reshape(obs['lidar'], (1, 1, -1, 1))   # (batch, t, lida, 1)
  angle = model(x)[0]
  action['steering'] = angle
  obs, rewards, done, states = env.step(action)
  t += 1
  returns += rewards
  if True:
    # Currently, two rendering modes are available: 'birds_eye' and 'follow'
    image = env.render(mode='birds_eye')
    video.append(image)
# store video
if True:
  # last frame
  image = env.render(mode='birds_eye')
  video.append(image)
  import imageio
  writer = imageio.get_writer(f'videos/ncp_treitlstrasse_{time.time()}.mp4', fps=100//8)
  for image in video:
      writer.append_data(image)
  writer.close()
# print out result
print(f"[Info] Cumulative Reward: {returns:.2f}")
print(f"[Info] Nr Sim Steps: {t * action_repeat}")
print(f"[Info] Simulated Time: {t * action_repeat / 100} seconds")
print(f"[Info] Real Time: {time.time() - init:.2f} seconds")
# close env
env.close()