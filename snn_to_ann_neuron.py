# Copyright (c) 2020 IBM Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
# Modified by Brian Onoszko and Noah Törnqvist on 2025-08-21:
# - Updated the code to work for Python version 3.10.11
# - Rewrote StepFunction to utilize quantization supported operators for STEdgeAI.
# - Rewrote call to utilize quanitzation supported operators for STEdgeAI.
# - Rewrote build to work with unrolling code
# - Removed StateTuple as it was no longer required.
# - Removed backward pass.
# - Removed recurrent spiking.

import tensorflow as tf

def StepFunction(x):
    """Mimics heaviside function using supported operators and returns either 0 or 1
    
    Keyword arguments:
        x -- membrane potential (Vm) + bias.
    """
    y = tf.clip_by_value(x * 0.5 + 0.5, 0.0, 1.0) 
    return tf.round(y)

class SNUCell(tf.keras.layers.Layer):
    def __init__(self, num_units, activation=StepFunction,
                 decay=0.8, initVth=0.0, **kwargs):
        
        """Initialize the SNUCell
    
        Keyword arguments:
            input_spec -- expected input dimention
            num_units -- numbers of nodes in layer
            activation -- activation function
            decay -- decay rate 
            initVth -- initial membrane potential
            kwargs -- misc. arguments passed to Keras base-class
        """
        
        super(SNUCell, self).__init__(**kwargs)
        self.input_spec = tf.keras.layers.InputSpec(ndim=2)
        self.num_units = num_units
        self.activation = activation
        self.decay_value = decay
        self.initVth = initVth
        # self.vm_log = [] # Log func

    @property
    def state_size(self):
        return [self.num_units, self.num_units]

    @property
    def output_size(self):
        return self.num_units

    def build(self, input_shape):
        """Builds layer weights based on the input shape.

        Keyword arguments:
        input_shape -- shape of the input tensor.
        """
        
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError(f"Wrong shape")
        self.kernel = self.add_weight(
            name="kernel", 
            shape=(input_dim, self.num_units)
        )
        self.bias = self.add_weight(
            shape=(self.num_units,),
            initializer=tf.keras.initializers.Constant(-self.initVth),
            name="bias"
        )
        self.decay = self.add_weight(
            shape=(self.num_units,),
            initializer=tf.keras.initializers.Constant(self.decay_value),
            trainable=False,
            name="decay"
        )
        super().build(input_shape)

    def call(self, inputs, states):
        """Executes one time step of the SNU cell. Computes updated membrane potential (Vm),
        output spike or activation and the next cell state.

        Keyword arguments:
        inputs -- input tensor at the current time step
        states -- states of membrane potential (Vm) and spike (h) from previous time step
        """
                
        Vm, h = states
        Vm = tf.multiply(Vm, tf.subtract(1.0, h))
        Vm = tf.multiply(Vm, self.decay)
        Vm = tf.add(tf.matmul(inputs, self.kernel), Vm)
        out = self.activation(Vm + self.bias)
        # self.vm_log.append(Vm) # Record Vm val
        return out, (Vm, out)