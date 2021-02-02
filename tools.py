import tensorflow as tf
import pathlib
import pickle
import numpy as np

class Module(tf.Module):

  def save(self, filename):
    values = tf.nest.map_structure(lambda x: x.numpy(), self.variables)
    with pathlib.Path(filename).open('wb') as f:
      pickle.dump(values, f)

  def load(self, filename):
    with pathlib.Path(filename).open('rb') as f:
      values = pickle.load(f)
    tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, values)

  def get(self, name, actor, *args, **kwargs):
    # Create or get layer by name to avoid mentioning it in the constructor.
    if not hasattr(self, '_modules'):
      self._modules = {}
    if name not in self._modules:
      self._modules[name] = actor(*args, **kwargs)
    return self._modules[name]

def write_video(video, filename, fps=100):
    import imageio
    writer = imageio.get_writer(filename, fps=fps)
    for image in video:
      writer.append_data(image)
    writer.close()


def plot_sample_model_predictions(model, data_x, data_y, axis, label, plot_true=False):
  for i, ax in enumerate(axis):
    pred = model(np.expand_dims(data_x[i], 0))[0]
    if plot_true:
      ax.plot(range(len(data_y[i])), data_y[i], label='true')
    ax.plot(range(len(data_y[i])), pred, label=label)
    ax.legend()