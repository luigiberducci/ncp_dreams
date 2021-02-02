import tools
import tensorflow.keras.layers as tfkl
import kerasncp as kncp
import tensorflow as tf


class ConvNCP(tools.Module):
    def __init__(self, name,
                 conv_filters, conv_kernels, conv_strides, encoded_dim,
                 inter_neurons, command_neurons, motor_neurons, sensory_fanout,
                 inter_fanout, recurrent_command_synapses, motor_fanin,
                 opt_lr):
        self._name = name
        self._loss_fn = tf.losses.MeanSquaredError()
        self._opt = tf.optimizers.Adam(opt_lr)
        self._head = ConvHead(filters=conv_filters, kernels=conv_kernels, strides=conv_strides, encoded_dim=encoded_dim)
        self._ncp = NCP(inter_neurons=inter_neurons, command_neurons=command_neurons, motor_neurons=motor_neurons,
                        sensory_fanout=sensory_fanout, inter_fanout=inter_fanout,
                        recurrent_command_synapses=recurrent_command_synapses, motor_fanin=motor_fanin)

    def __call__(self, inputs, **kwargs):
        embed_reshape = tf.concat([tf.shape(inputs)[:-2], [32]], 0)
        x = tf.reshape(inputs, (-1,) + tuple(inputs.shape[-2:]))  # apply same conv to all B*T observations
        embeds = self._head(x)
        embeds = tf.reshape(embeds, embed_reshape)
        return self._ncp(embeds)  # rnn layer takes input of shape (B, T, *)

    def evaluate(self, validation_dataset):
        losses = []
        for x_batch_val, y_batch_val in validation_dataset:
            pred = self(x_batch_val)
            loss_value = self._loss_fn(y_batch_val, pred)
            losses.append(loss_value)
        return tf.reduce_mean(losses)

    def build_model(self):
        dummy_input = tf.zeros([1, 1, 1080, 1])
        self(dummy_input)

    @tf.function
    def train_step(self, x_batch_train, y_batch_train):
        with tf.GradientTape() as tape:
            pred = self(x_batch_train)
            loss_value = self._loss_fn(y_batch_train, pred)
        grads = tape.gradient(loss_value, self.variables)
        self._opt.apply_gradients(zip(grads, self.variables))
        return loss_value


class ConvAutoEncoderNCP(tools.Module):
    def __init__(self, name,
                 conv_filters, conv_kernels, conv_strides, encoded_dim,
                 inter_neurons, command_neurons, motor_neurons, sensory_fanout,
                 inter_fanout, recurrent_command_synapses, motor_fanin,
                 opt_lr):
        self._name = name
        self._loss_fn = tf.losses.MeanSquaredError()
        self._opt = tf.optimizers.Adam(opt_lr)
        self._encoder = ConvEncoder(filters=conv_filters, kernels=conv_kernels, strides=conv_strides,
                                encoded_dim=encoded_dim)
        self._decoder = ConvDecoder(filters=conv_filters, kernels=conv_kernels, strides=conv_strides,
                                encoded_dim=encoded_dim)
        self._ncp = NCP(inter_neurons=inter_neurons, command_neurons=command_neurons, motor_neurons=motor_neurons,
                        sensory_fanout=sensory_fanout, inter_fanout=inter_fanout,
                        recurrent_command_synapses=recurrent_command_synapses, motor_fanin=motor_fanin)

    def __call__(self, inputs, **kwargs):
        embed_reshape = tf.concat([tf.shape(inputs)[:-2], [32]], 0)
        x = tf.reshape(inputs, (-1,) + tuple(inputs.shape[-2:]))  # apply same conv to all B*T observations
        embeds = self._head(x)
        embeds = tf.reshape(embeds, embed_reshape)
        return self._ncp(embeds)  # rnn layer takes input of shape (B, T, *)

    def evaluate(self, validation_dataset):
        losses = []
        for x_batch_val, y_batch_val in validation_dataset:
            pred = self(x_batch_val)
            loss_value = self._loss_fn(y_batch_val, pred)
            losses.append(loss_value)
        return tf.reduce_mean(losses)

    def build_model(self):
        dummy_input = tf.zeros([1, 1, 1080, 1])
        self(dummy_input)

    @tf.function
    def train_step(self, x_batch_train, y_batch_train):
        with tf.GradientTape() as tape:
            pred = self(x_batch_train)
            loss_value = self._loss_fn(y_batch_train, pred)
        grads = tape.gradient(loss_value, self.variables)
        self._opt.apply_gradients(zip(grads, self.variables))
        return loss_value


class SteeringImitationModel(ConvNCP):
    def __init__(self, name, conv_filters=[18, 20, 22, 24, 25], conv_kernels=[10, 10, 10, 5, 5],
                 conv_strides=[3, 2, 2, 1, 1], encoded_dim=32,
                 inter_neurons=12, command_neurons=19, motor_neurons=1, sensory_fanout=6,
                 inter_fanout=4, recurrent_command_synapses=6, motor_fanin=4, opt_lr=1e-3):
        super(SteeringImitationModel, self).__init__(name, conv_filters, conv_kernels, conv_strides, encoded_dim,
                                                     inter_neurons, command_neurons, motor_neurons, sensory_fanout,
                                                     inter_fanout, recurrent_command_synapses, motor_fanin, opt_lr)


class MotorSteeringImitationModel(ConvNCP):
    def __init__(self, name, conv_filters=[18, 20, 22, 24, 25], conv_kernels=[10, 10, 10, 5, 5],
                 conv_strides=[3, 2, 2, 1, 1], encoded_dim=32,
                 inter_neurons=12, command_neurons=19, motor_neurons=2, sensory_fanout=6,
                 inter_fanout=4, recurrent_command_synapses=6, motor_fanin=4, opt_lr=1e-3):
        super(MotorSteeringImitationModel, self).__init__(name, conv_filters, conv_kernels, conv_strides, encoded_dim,
                                                          inter_neurons, command_neurons, motor_neurons, sensory_fanout,
                                                          inter_fanout, recurrent_command_synapses, motor_fanin, opt_lr)


class NCP(tools.Module):

    def __init__(self, inter_neurons=12, command_neurons=19, motor_neurons=1,
                 sensory_fanout=6, inter_fanout=4, recurrent_command_synapses=6, motor_fanin=4):
        self._name = 'ncp_layer'
        ncp_arch = kncp.wirings.NCP(
            inter_neurons=inter_neurons,  # Number of inter neurons
            command_neurons=command_neurons,  # Number of command neurons
            motor_neurons=motor_neurons,  # Number of motor neurons
            sensory_fanout=sensory_fanout,  # How many outgoing synapses has each sensory neuron
            inter_fanout=inter_fanout,  # How many outgoing synapses has each inter neuron
            recurrent_command_synapses=recurrent_command_synapses,
            # Now many recurrent synapses are in the command neuron layer
            motor_fanin=motor_fanin,  # How many incomming syanpses has each motor neuron
        )
        self._ncp_cell = tfkl.RNN(kncp.LTCCell(ncp_arch), return_sequences=True)

    def __call__(self, inputs, **kwargs):
        return self._ncp_cell(inputs, **kwargs)


class ConvHead(tools.Module):
    def __init__(self, filters=[18, 20, 22, 24, 26], kernels=[10, 10, 10, 5, 5], strides=[3, 2, 2, 1, 1],
                 encoded_dim=32, activation='relu'):
        self._name = 'conv_head'
        self._encoded_dim = encoded_dim
        self._act = activation
        self._filters = filters
        self._kernels = kernels
        self._strides = strides

    def __call__(self, x):
        for i, (filters, kernel, stride) in enumerate(zip(self._filters, self._kernels, self._strides)):
            x = self.get(f'conv{i + 1}', tfkl.Conv1D, filters=filters, kernel_size=kernel, strides=stride,
                         activation=self._act)(x)
            if i > 0 and i % 2 == 1:  # every 2 conv layers, put a max pool layer
                x = self.get(f'max-pool{i}', tfkl.MaxPool1D)(x)
        x = self.get('flat', tfkl.Flatten)(x)
        x = self.get('dense', tfkl.Dense, units=self._encoded_dim, activation=self._act)(x)
        return x


class ConvEncoder(tools.Module):
    def __init__(self, filters=[18, 20, 22, 24, 26], kernels=[10, 10, 10, 5, 5], strides=[3, 2, 2, 1, 1],
                 encoded_dim=32, activation='relu'):
        self._name = 'conv_head'
        self._encoded_dim = encoded_dim
        self._act = activation
        self._filters = filters
        self._kernels = kernels
        self._strides = strides

    def __call__(self, x):
        x = self.get(f'conv1', tfkl.Conv1D, filters=18, kernel_size=10, strides=3, activation=self._act)(x)
        x = self.get(f'conv2', tfkl.Conv1D, filters=20, kernel_size=10, strides=2, activation=self._act)(x)
        x = self.get(f'max-pool1', tfkl.MaxPool1D)(x)
        x = self.get(f'conv3', tfkl.Conv1D, filters=22, kernel_size=10, strides=2, activation=self._act)(x)
        x = self.get(f'conv4', tfkl.Conv1D, filters=24, kernel_size=5, strides=1, activation=self._act)(x)
        x = self.get(f'max-pool2', tfkl.MaxPool1D)(x)
        x = self.get(f'conv5', tfkl.Conv1D, filters=26, kernel_size=5, strides=1, activation=self._act)(x)
        x = self.get('flat', tfkl.Flatten)(x)
        x = self.get('dense', tfkl.Dense, units=self._encoded_dim, activation=self._act)(x)
        return x


class ConvDecoder(tools.Module):
    def __init__(self, filters=[18, 20, 22, 24, 26], kernels=[10, 10, 10, 5, 5], strides=[3, 2, 2, 1, 1],
                 encoded_dim=32, activation='relu'):
        self._name = 'conv_head'
        self._encoded_dim = encoded_dim
        self._act = activation
        self._filters = filters
        self._kernels = kernels
        self._strides = strides

    def __call__(self, x):
        x = self.get(f'conv1', tfkl.Conv1DTranspose, filters=26, kernel_size=5, strides=1, activation=self._act)(x)
        x = self.get(f'conv2', tfkl.Conv1DTranspose, filters=24, kernel_size=5, strides=1, activation=self._act)(x)
        x = self.get(f'max-pool1', tfkl.MaxPool1D)(x)
        x = self.get(f'conv3', tfkl.Conv1DTranspose, filters=22, kernel_size=10, strides=2, activation=self._act)(x)
        x = self.get(f'conv4', tfkl.Conv1DTranspose, filters=20, kernel_size=10, strides=2, activation=self._act)(x)
        x = self.get(f'max-pool2', tfkl.MaxPool1D)(x)
        x = self.get(f'conv5', tfkl.Conv1DTranspose, filters=18, kernel_size=10, strides=3, activation=self._act)(x)
        x = self.get('flat', tfkl.Flatten)(x)
        x = self.get('dense', tfkl.Dense, units=self._encoded_dim, activation=self._act)(x)
        return x
