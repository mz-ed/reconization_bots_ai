import os
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


CAT_DB = "train_transformed/cats"
DOG_DB = "train_transformed/dogs"

class NeuronLayer(tf.Module):
    def __init__(self, num_inputs, num_neurons, name=   None):
        super().__init__(name=name)
        self.weights = tf.Variable(tf.random.normal([num_inputs, num_neurons]), name="weights")
        self.bias = tf.Variable(tf.random.normal([1, num_neurons]), name="bias")

    def forward(self, inputs):
        linear_output = tf.add(tf.matmul(inputs, self.weights), self.bias)
        return tf.nn.sigmoid(linear_output)

# Load and preprocess images
def load_images_from_folder(folder, label, size=(64, 64)):
    images = []
    labels = []
    paths = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path).resize(size)
                img_array = np.array(img).astype(np.float32) / 255.0
                img_flat = img_array.reshape(-1)
                images.append(img_flat)
                labels.append(label)
                paths.append(img_path)
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
    return images, labels, paths

# Load dataset

cat_images, cat_labels, cat_paths = load_images_from_folder(CAT_DB, label=0)
dog_images, dog_labels, dog_paths = load_images_from_folder(DOG_DB, label=1)

X = np.array(cat_images + dog_images)
y = np.array(cat_labels + dog_labels).reshape(-1, 1)
image_paths = cat_paths + dog_paths

indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
image_paths = [image_paths[i] for i in indices]

# Convert to tensors
X_tensor = tf.constant(X, dtype=tf.float32)
Y_tensor = tf.constant(y, dtype=tf.float32)



# Fake labels (all cats = 1.0)
y_true = tf.ones((Y_tensor.shape[0], 1), dtype=tf.float32)

# Create the neural network
layer1 = NeuronLayer(num_inputs=X_tensor.shape[1], num_neurons=128)
layer2 = NeuronLayer(num_inputs=128, num_neurons=64)
layer3 = NeuronLayer(num_inputs=64, num_neurons=32)
layer4 = NeuronLayer(num_inputs=32, num_neurons=16)
layer5 = NeuronLayer(num_inputs=16, num_neurons=1)

# Training parameters
learning_rate = 0.001
epochs = 250

# Use Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate)

loss_history = []
# Training loop
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        out1 = layer1.forward(X_tensor)
        out2 = layer2.forward(out1)
        out3 = layer3.forward(out2)
        out4 = layer4.forward(out3)
        out5 = layer5.forward(out4)
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(Y_tensor, out5))

    gradients = tape.gradient(loss, [layer1.weights, layer1.bias,
                                     layer2.weights, layer2.bias,
                                     layer3.weights, layer3.bias,
                                     layer4.weights, layer4.bias,
                                     layer5.weights, layer5.bias])
    optimizer.apply_gradients(zip(gradients, [layer1.weights, layer1.bias,
                                              layer2.weights, layer2.bias,
                                              layer3.weights, layer3.bias,
                                              layer4.weights, layer4.bias,
                                              layer5.weights, layer5.bias]))

    loss_history.append(loss.numpy())
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.numpy():.4f}")
        predictions = tf.cast(out5 > 0.5, tf.float32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, Y_tensor), tf.float32))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.numpy():.4f}, Accuracy: {accuracy.numpy():.4f}")

ckpt = tf.train.Checkpoint(
    layer1=layer1,
    layer2=layer2,
    layer3=layer3,
    layer4=layer4,
    layer5=layer5,
  
  
)

ckpt.write("model_catvsdog_checkpoint.ckpt")

def predict_image(image_path, model_layers, image_size=(64, 64)):
    try:
        # Load and preprocess image correctly
        img = Image.open(image_path).convert("RGB").resize(image_size)  # Ensure RGB and resize
        img_array = np.array(img).astype(np.float32) / 255.0
        img_flat = img_array.reshape(1, -1)  # Flatten into shape [1, 12288]

        # Check input shape
        expected_input_size = model_layers[0].weights.shape[0]
        if img_flat.shape[1] != expected_input_size:
            raise ValueError(f"Image input size mismatch: got {img_flat.shape[1]}, expected {expected_input_size}")

        # Tensor conversion
        img_tensor = tf.constant(img_flat, dtype=tf.float32)

        # Forward pass
        out = img_tensor
        for layer in model_layers:
            out = layer.forward(out)

        prediction = out.numpy()[0][0]
        label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
        print(f"\nPrediction: {label} (Confidence: {prediction:.4f})")

        # Show image with prediction
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{label} ({prediction:.2f})")
        plt.show()

    except Exception as e:
        print(f"Error loading image: {e}")

# Final output after training
final_output =  layer5.forward(layer4.forward(layer3.forward(layer2.forward(layer1.forward(X_tensor)))))


# Plot histogram of final outputs
plt.figure(figsize=(8, 5))
plt.hist(final_output.numpy(), bins=30, color='skyblue', edgecolor='black')
plt.title("Final Output Distribution (After Training)")
plt.xlabel("Output Value (Sigmoid)")
plt.ylabel("Number of Images")
plt.grid(True)
plt.show()

# --- Top 5 Most Cat-Like Predictions ---
top_indices = tf.argsort(tf.reshape(final_output, [-1]), direction='DESCENDING')[:5].numpy()

print("\nTop 5 most 'dog-like' images (highest network output):")
plt.figure(figsize=(15, 5))
for i, idx in enumerate(top_indices):
    img = Image.open(image_paths[idx])
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Score: {final_output[idx][0]:.2f}")
plt.suptitle("Top 5 Most Dog-Like Images (According to Network)", fontsize=16)
plt.show()

trained_layers = [layer1, layer2, layer3, layer4, layer5]

# Path to your test image
test_path = "pngtree-isolated-cat-on-white-background-png-image_9158356.png"

# Run prediction
predict_image(test_path, trained_layers)

