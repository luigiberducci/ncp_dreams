import gym
import wrappers
import numpy as np

task = 'SingleAgentTreitlstrasse_v2-v0'
time_limit = 60*100
action_repeat = 8
env = gym.make(task)
env = wrappers.TimeLimit(env, time_limit)
env = wrappers.ActionRepeat(env, action_repeat)


def test_on_track(model, outdir):
    video, returns = simulate_episode(model)
    videodir = outdir / 'videos'
    videodir.mkdir(parents=True, exist_ok=True)
    import imageio
    writer = imageio.get_writer(videodir / f'test_return{returns}.mp4')
    for image in video:
        writer.append_data(image)
    writer.close()


def simulate_episode(model, prediction_window=5, terminate_on_collision=True):
    # to do: make it uniform to f1_tenth directory
    done = False
    obs = env.reset(mode='grid')
    state = None
    video = []
    returns = 0.0
    while not done:
        action = {}
        x = obs['lidar'] / 15.0 - 0.5
        x = np.reshape(x, (1, 1, -1, 1)).astype(np.float32)  # (batch, t, lida, 1)
        if state is None:
            state = x
        else:
            state = np.concatenate([state, x], axis=1)[:, -prediction_window:, :, :]  # concatenate over time axis
        a = model(state)[0, -1]  # take last action of the sequence
        motor, steering = max(0.005, a[0]), a[1]  # avoid agent stays still
        action['motor'] = motor
        action['steering'] = steering
        obs, rewards, done, states = env.step(action)
        if terminate_on_collision and states['wall_collision']:
            done = True
        returns += rewards
        image = env.render(mode='birds_eye')
        video.append(image)
    image = env.render(mode='birds_eye')
    video.append(image)
    return video, returns
