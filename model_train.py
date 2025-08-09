import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from snn_to_ann_unroll import SNUCell, UnrolledSNU

# 1. PARAMETERS
time_steps = 10
num_hidden = 100
num_hidden2 = 0
num_digits = 10
batch_size = 1000
epochs = 5
learning_rate = 1e-3

tflite_model_path = 'snn_100_1.tflite'


# 2. RATE-ENCODING FUNCTION
def rate_encode(samples, time_steps):
    batch, features = samples.shape
    rnd  = np.random.rand(batch, time_steps, features)
    inten = np.repeat(samples[:, None, :], time_steps, axis=1)
    spikes = (rnd < inten).astype(np.float32)
    return spikes

def representative_dataset_gen():
    for i in range(batch_size):
        sample = x_train[i : i + 1] 
        spikes = rate_encode(sample, time_steps)
        yield [spikes]

# 3. LOAD MNIST AND PREPARE SPIKE-TRAIN ARRAYS
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

spike_train = rate_encode(x_train, time_steps)
spike_test = rate_encode(x_test,  time_steps)

labels_train = tf.one_hot(y_train, depth=num_digits, dtype=tf.float32)
labels_test = tf.one_hot(y_test, depth=num_digits, dtype=tf.float32)

# 4. DEFINE CONFUSION‐MATRIX CALLBACK
class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.x_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs=None):
        y_pred_prob = self.model.predict(self.x_val, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(self.y_val, axis=1)

        cm_tensor = tf.math.confusion_matrix(
            labels=y_true,
            predictions=y_pred,
            num_classes=num_digits,
            dtype=tf.int32
        )
        cm = cm_tensor.numpy()

        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest')
        plt.title(f"Confusion Matrix - Epoch {epoch + 1}")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.colorbar()

        ticks = np.arange(num_digits)
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.gca().set_xticks(ticks - 0.5, minor=True)
        plt.gca().set_yticks(ticks - 0.5, minor=True)
        plt.gca().grid(which="minor", color="w", linestyle='-', linewidth=1)
        plt.tight_layout()
        plt.show()

# 5. BUILD THE MODEL
inputs = tf.keras.Input(shape=(time_steps, 784), dtype=tf.float32)

cell = SNUCell(num_units=num_hidden)
snn_layer = UnrolledSNU(cell, time_steps)(inputs)

x = tf.keras.layers.Lambda(lambda y: tf.reduce_sum(y, axis=1))(snn_layer)
dense = tf.keras.layers.Dense(num_digits, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=dense)

# 6. COMPILE
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)

# 7. TRAIN 
confusion_cb = ConfusionMatrixCallback(validation_data=(spike_test, labels_test))

history = model.fit(
    x=spike_train,
    y=labels_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(spike_test, labels_test),
    callbacks=[confusion_cb]
)

# 8. FINAL EVALUATION
test_loss, test_acc = model.evaluate(spike_test, labels_test, batch_size=512)
print(f"\nFinal test accuracy on rate-coded MNIST: {test_acc:.4f}")

# 9. SAVE MODEL FOR LATER EXPORT
path = "snn_mnist_trained_savedmodel"
model.export(path)
print("Trained model saved to 'snn_mnist_trained_savedmodel'")

# 10. CONVERT TO TFLITE FORMAT 

converter = tf.lite.TFLiteConverter.from_saved_model(path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset_gen

converter.experimental_new_quantizer = True

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = [tf.int8]
tf.compat.v1.enable_resource_variables()
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.uint8

tflite_quant_model = converter.convert()
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_quant_model)

print(f"Quantized TFLite model saved to {tflite_model_path}")