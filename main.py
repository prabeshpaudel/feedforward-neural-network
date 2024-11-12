from neuralnet import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

learning_rates = {
    "relu": 0.001,
    "sigmoid": 0.1,
    "tanh": 0.01,
    "leaky_relu": 0.001
}

# Set parameters and create a neural network instance
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
activation_function = 'relu'
learning_rate = learning_rates[activation_function]
nn_model = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, activation_function)

# Load the MNIST dataset from TensorFlow
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Flatten and normalize the images
train_images = train_images.reshape(train_images.shape[0], 784) / 255.0 * 0.99 + 0.01
test_images = test_images.reshape(test_images.shape[0], 784) / 255.0 * 0.99 + 0.01

# Visualize a sample from the MNIST training dataset
index = 0  # specify which sample to display
image_array = train_images[index].reshape((28, 28))
# plt.imshow(image_array, cmap='Greys', interpolation='none')
# plt.show()
# print("The target value is:", train_labels[index])

# Train the neural network over multiple epochs
epochs = 10
for e in range(epochs):
    print("Epoch:", e + 1, "/", epochs)
    for i in range(len(train_images)):
        inputs = train_images[i]
        targets = np.zeros(output_nodes) + 0.01
        targets[train_labels[i]] = 0.99
        nn_model.train(inputs, targets)

# Evaluate the network's performance
scorecard = []
for i in range(len(test_images)):
    correct_label = test_labels[i]
    inputs = test_images[i]
    outputs = nn_model.predict(inputs)
    label = np.argmax(outputs)
    scorecard.append(1 if label == correct_label else 0)

# Calculate and print performance
scorecard_array = np.asarray(scorecard)
print("Performance =", (scorecard_array.sum() / scorecard_array.size) * 100, "%")
