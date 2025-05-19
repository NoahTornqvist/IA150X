import tensorflow as tf
import collections
from tensorflow.keras.layers import Layer, InputSpec

print("TensorFlow version:", tf.__version__)

def StepFunction(x):
    return tf.nn.tanh(x) + tf.stop_gradient(-tf.nn.tanh(x) + tf.nn.relu(tf.sign(x)))

# Define the state tuple
SNUCellTuple = collections.namedtuple("SNUCellStateTuple", ("Vm", "h"))
class SNUCellStateTuple(SNUCellTuple):
    __slots__ = ()
    @property
    def dtyp(self):
        (Vm, h) = self
        if Vm.dtype != h.dtype:
            raise TypeError("Inconsistent internal state")
        return Vm.dtype

class SNUCell(tf.keras.layers.Layer):
    def __init__(self, num_units, activation=StepFunction, g=tf.nn.relu,
                 decay=0.8, initVth=1.0, **kwargs):
        super(SNUCell, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=2)
        self.num_units = num_units
        self.activation = activation
        self.g = g
        self.decay_value = decay
        self.initVth = initVth
        self.vm_log = [] # Log func

    @property
    def state_size(self):
        return [self.num_units, self.num_units]

    @property
    def output_size(self):
        return self.num_units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.num_units),
            initializer=tf.constant_initializer(1.0),
            name="kernel"
        )
        self.bias = self.add_weight(
            shape=(self.num_units,),
            initializer=tf.constant_initializer(-self.initVth),
            name="bias"
        )
        self.decay = self.add_weight(
            shape=(self.num_units,),
            initializer=tf.constant_initializer(self.decay_value),
            trainable=False,
            name="decay"
        )
        super().build(input_shape)

    def call(self, inputs, states):
        Vm, h = states
        Vm = tf.multiply(Vm, tf.subtract(1.0,h))
        Vm = tf.multiply(Vm, self.decay)
        Vm = tf.add(tf.matmul(inputs, self.kernel), Vm)
        Vm = self.g(Vm)
        out = self.activation(Vm + self.bias)
        self.vm_log.append(Vm) # Record Vm val
        return out, SNUCellStateTuple(Vm, out)
    
def main():
    batch_size = 1
    timesteps = 5
    input_dim = 3
    units = 4

    # Dummy input: (batch, time, features)
    x = tf.random.normal((batch_size, timesteps, input_dim))

    # Wrap the cell into a Keras RNN layer
    cell = SNUCell(units)
    rnn_layer = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)

    # Run
    output, final_Vm, final_h = rnn_layer(x)
    print("Output:", output)
    print("Final Vm:", final_Vm)
    print("Final h:", final_h)

if __name__ == "__main__":
    main()