import kerasncp as kncp
import tensorflow.keras as keras
import tensorflow.keras.layers as kl


class NCP(keras.Model):

    def __init__(self):
        super(NCP, self).__init__()
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
        self.input_layer = kl.InputLayer(input_shape=(None, 2))
        self.dense_1 = kl.Dense(128, activation='relu')
        self.dense_2 = kl.Dense(32, activation='linear')
        self.ncp = kl.RNN(ncp_cell, return_sequences=False)

    def call(self, inputs, **kwargs):
        x = self.input_layer(inputs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.ncp(x)
        return x

    def get_config(self):
        pass
