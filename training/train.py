import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import time

@tf.function
def train_step(model, x_batch_train, y_batch_train, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        pred = model(x_batch_train)
        loss_value = loss_fn(y_batch_train, pred)
    grads = tape.gradient(loss_value, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables))
    return loss_value


def train_loop(model, train_dataset, val_dataset, epochs, writer, hparams=None):
    with writer.as_default():
        hp.hparams(hparams)
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
                print(f"\tBatch {step + 1}: train loss: {train_loss.result():.5f}")
        val_loss.update_state(model.evaluate(val_dataset))
        with writer.as_default():
            tf.summary.scalar("train_loss", train_loss.result(), step=epoch)
            tf.summary.scalar("val_loss", val_loss.result(), step=epoch)

        print(f'Epoch {epoch + 1}: train loss: {train_loss.result():.5f}, ' \
              f'validation loss: {val_loss.result():.5f}, time: {time.time() - init:.5f} sec')
    return model