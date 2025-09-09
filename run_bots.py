import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class NeuronLayer(tf.Module):
    def __init__(self, num_inputs, num_neurons, name=None):
        super().__init__(name=name)
        self.weights = tf.Variable(tf.random.normal([num_inputs, num_neurons]), name="weights")
        self.bias = tf.Variable(tf.random.normal([1, num_neurons]), name="bias")

    def forward(self, inputs):
        linear_inputs = tf.add(tf.matmul(inputs, self.weights), self.bias)
        return tf.sigmoid(linear_inputs)

def predict_image(image_path, model_layers, image_size=(64, 64)):
    try:
        img = Image.open(image_path).convert("RGB").resize(image_size)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_flat = img_array.reshape(1, -1)

        expected_input_size = model_layers[0].weights.shape[0]
        if img_flat.shape[1] != expected_input_size:
            raise ValueError(f"Image input size mismatch: got {img_flat.shape[1]}, expected {expected_input_size}")

        img_tensor = tf.constant(img_flat, dtype=tf.float32)
        out = img_tensor
        for layer in model_layers:
            out = layer.forward(out)

        prediction = out.numpy()[0][0]
        label = "woman" if prediction > 0.5 else "man"
        print(f"\nPrediction: {label} (Confidence: {prediction:.4f})")

        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{label} ({prediction:.2f})")
        plt.show()

    except Exception as e:
        print(f"Error loading image: {e}")

# === Rebuild Model ===
INPUT_SIZE = 64 * 64 * 3

layer1 = NeuronLayer(num_inputs=INPUT_SIZE, num_neurons=256)
layer2 = NeuronLayer(num_inputs=256, num_neurons=128)
layer3 = NeuronLayer(num_inputs=128, num_neurons=64)
layer4 = NeuronLayer(num_inputs=64, num_neurons=32)
layer5 = NeuronLayer(num_inputs=32, num_neurons=16)
layer6 = NeuronLayer(num_inputs=16, num_neurons=1)

model_layers = [layer1, layer2, layer3, layer4, layer5, layer6]

# === Load Checkpoint ===
ckpt = tf.train.Checkpoint(
    layer1=layer1,
    layer2=layer2,
    layer3=layer3,
    layer4=layer4,
    layer5=layer5,
    layer6=layer6
)
ckpt.restore("model_checkpoint.ckpt").expect_partial()

# === Predict on image ===
predict_image("5888934070656815654.jpg", model_layers)
