import tensorflow as tf
import matplotlib.pyplot as plt
from snn_to_ann_neuron import *

# Assume SNUCell is already defined as shown previously
# Create a dummy input: zero input over T timesteps
timesteps = 50
input_dim = 1
batch_size = 1

# Convert to a TensorFlow tensor of shape (1, 50, 1)
inputs = tf.constant(
    [[
        [0.0], [0.0], [0.0], [0.0], [1.0],
        [0.0], [0.0], [1.0], [0.0], [0.0],
        [0.0], [0.0], [0.0], [0.0], [0.0],
        [0.0], [0.0], [0.0], [0.0], [0.0],
        [0.0], [0.0], [0.0], [0.0], [0.0],
        [0.0], [0.0], [0.0], [0.0], [0.0],
        [0.0], [0.0], [0.0], [0.0], [0.0],
        [0.0], [0.0], [0.0], [0.0], [0.0],
        [0.0], [0.0], [0.0], [0.0], [0.0],
        [0.0], [0.0], [0.0], [0.0], [0.0]
    ]],
    dtype=tf.float32
)
inputs = tf.reshape(inputs, (batch_size, timesteps, input_dim))

initial_voltage = 1.0
cell = SNUCell(num_units=1)

# Define the RNN layer with return_sequences to get the voltage over time
rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)

initial_state = [tf.constant([[initial_voltage]], dtype=tf.float32),
                 tf.zeros((1, 1), dtype=tf.float32)]

# Run the RNN
outputs, Vm_final, h_final = rnn(inputs, initial_state=initial_state)

# Collect Vm values from the cell's internal log
vm_values = tf.stack(cell.vm_log, axis=1).numpy().squeeze()

# Extract h values
h_values = outputs.numpy().squeeze()

print("======= RESULT =======")
for i in range(timesteps):
    print(f"time-step {i}: {cell.vm_log[i]}")

plt.figure(figsize=(10, 4))
plt.plot(vm_values, marker='o', label='Vm (membrane potential)')
plt.plot(h_values, marker='x', linestyle='--', label='h (spike output)')
plt.title("Vm and h Over Time")
plt.xlabel("Time step")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.show()

