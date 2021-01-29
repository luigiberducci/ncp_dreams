import numpy as np
import os
from tensorflow import keras
import kerasncp as kncp
import matplotlib.pyplot as plt
import seaborn as sns
from models.ncp_model import NCP
import tensorflow as tf
import tensorflow.keras.optimizers.Adam as Adam

# load data

# define model
mime = NCP()
optimizer = Adam(learning_rate=1e-3)

mime.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
mime.summary()

# train model
hist = mime.fit(x_train, y_train, epochs=400, verbose=1)

# plot loss
sns.set()
plt.figure(figsize=(6, 4))
plt.plot(hist.history["loss"], label="Training loss")
plt.legend(loc="upper right")
plt.xlabel("Training steps")
plt.show()
