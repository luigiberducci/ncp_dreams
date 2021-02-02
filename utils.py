from typing import List

import numpy as np
import pathlib


def load_episodes(path: pathlib.Path, motor_neurons: int):
    x, y = [], []
    for file in path.glob('*.npy'):
        observations = []
        actions = []
        data = np.load(file, allow_pickle=True)
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


def create_sequences(data_x, data_y, length=50):
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


def plot_sample_model_predictions(model, data_x, data_y, axis, label, plot_true=False):
    for i, ax in enumerate(axis):
        pred = model(np.expand_dims(data_x[i], 0))[0]
        if plot_true:
            ax.plot(range(len(data_y[i])), data_y[i], label='true')
        ax.plot(range(len(data_y[i])), pred, label=label)
        ax.legend()


def simulate_episode(model, motor_neurons, action_repeat=8, rendering=True):
    import gym
    import wrappers

    task = 'SingleAgentTreitlstrasse_v2_Gui-v0' if rendering else 'SingleAgentTreitlstrasse_v2-v0'
    env = gym.make(task)
    env = wrappers.TimeLimit(env, duration=60 * 100)
    env = wrappers.ActionRepeat(env, action_repeat)

    done = False
    obs = env.reset(mode='grid')
    state = None
    video = []
    returns = 0.0
    while not done:
        action = {}
        x = np.reshape(obs['lidar'], (1, 1, -1, 1)).astype(np.float32)  # (batch, t, lida, 1)
        if state is None:
            state = x
        else:
            state = np.concatenate([state, x], axis=1)  # concatenate over time axis
        if motor_neurons == 1:
            motor, steering = 0.01, model(state)[0, -1, :]      # take last action of the sequence
        else:
            a = model(state)[0, 0, -1, :]                       # take last action of the sequence
            motor, steering = a[0], a[1]
        action['motor'] = motor
        action['steering'] = steering
        obs, rewards, done, states = env.step(action)
        returns += rewards
        image = env.render(mode='birds_eye')
        video.append(image)
    image = env.render(mode='birds_eye')
    video.append(image)
    env.close()
    return video, returns


def write_video(video, filename, fps=100):
    import imageio
    writer = imageio.get_writer(filename, fps=fps)
    for image in video:
        writer.append_data(image)
    writer.close()
