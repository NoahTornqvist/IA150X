import tensorflow as tf
from snn_to_ann_neuron import *

class UnrolledSNU(tf.keras.layers.Layer):
    def __init__(self, cell, time_steps, **kwargs):
        """Initializes the inner cell and the unroll sequence
    
        Keyword arguments:
            cell -- the SNU cell
            time_steps -- number of time steps
            kwargs -- misc. arguments passed to Keras base-class
        """
                
        super(UnrolledSNU, self).__init__(**kwargs)
        self.cell = cell
        self.time_steps = time_steps

    def build(self, input_shape):
        """Builds the inner cell
    
        Keyword arguments:
            input_shape -- shape of the input tensor
        """

        self.cell.build(input_shape[1:])  # shape = (batch, time, features)
        super().build(input_shape)

    def call(self, inputs):
        """Processes a sequence of inputs through the SNU cell.
    
        Keyword arguments:
            inputs -- input tensor of shape [batch_size, time_steps, features]
        """
                
        batch_size = tf.shape(inputs)[0]
        outputs = []
        
        h = tf.zeros((batch_size, self.cell.num_units), dtype=tf.float32)
        Vm = tf.zeros((batch_size, self.cell.num_units), dtype=tf.float32)

        for t in range(self.time_steps):
            x_t_uint8 = inputs[:, t, :]
            x_t = tf.cast(x_t_uint8, tf.float32)
            out, state = self.cell(x_t, [Vm, h])
            Vm, h = state
            outputs.append(tf.expand_dims(out, axis=1))

        return tf.concat(outputs, axis=1)