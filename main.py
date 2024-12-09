from neuralnet import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_auc_score

# Define learning rates for different activation functions
learning_rates = {
    "relu": 0.001,
    "sigmoid": 0.1,
    "tanh": 0.01,
    "leaky_relu": 0.001
}

# List of activation functions and weight initialization methods to test
activation_functions = ['sigmoid', 'tanh', 'relu', 'leaky_relu']
weight_initializations = ['random', 'xavier', 'he']
regularizations = {
    "l2_lambda": [0.001, 0.01, 0.1],
    "dropout_rate": [0.2, 0.4, 0.6]
}

# Set parameters for the neural network
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
epochs = 10  # Reduced number of epochs for quicker execution during testing

# Load the MNIST dataset from TensorFlow
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Flatten the images into vectors
train_images = train_images.reshape(train_images.shape[0], 784)
test_images = test_images.reshape(test_images.shape[0], 784)


# Function to normalize images based on activation function requirements
def normalize_images(images, activation_function):
    if activation_function == 'sigmoid':
        # Scale to [0.01, 1.0] for sigmoid activation
        return images / 255.0 * 0.99 + 0.01
    elif activation_function == 'tanh':
        # Scale to [-1.0, 1.0] for tanh activation
        return (images / 255.0 - 0.5) * 2
    elif activation_function in ['relu', 'leaky_relu']:
        # Scale to [0.0, 1.0] for ReLU and Leaky ReLU activation
        return images / 255.0
    else:
        return images


# Function to calculate AUC for multi-class classification
def compute_auc(model, test_images, test_labels):
    all_true_labels = np.zeros((len(test_labels), 10))  # Assuming 10 classes in MNIST
    all_predictions = np.zeros_like(all_true_labels)

    # Convert test_labels to one-hot encoding
    for i, label in enumerate(test_labels):
        all_true_labels[i, label] = 1

    for i in range(len(test_images)):
        inputs = test_images[i]
        outputs = model.predict(inputs)
        all_predictions[i] = outputs.T  # Transpose to match dimensions

    # Compute AUC for each class and average them
    auc_score = roc_auc_score(all_true_labels, all_predictions, average='macro', multi_class='ovr')

    return auc_score


# Visualize a sample from the MNIST training dataset
index = 0  # specify which sample to display
image_array = train_images[index].reshape((28, 28))
# Uncomment the following lines to display the image
# plt.imshow(image_array, cmap='Greys', interpolation='none')
# plt.show()
print("The target value is:", train_labels[index])


# Train and evaluate function
def train_and_evaluate(activation_function, weight_init, reg_type=None, reg_value=None):
    # Normalize images
    train_inputs = normalize_images(train_images, activation_function)
    test_inputs = normalize_images(test_images, activation_function)

    # Set learning rate based on activation function
    learning_rate = learning_rates[activation_function]

    # Create neural network model with or without regularization
    if reg_type == "l2_lambda":
        nn_model = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate,
                                 activation_function, weight_init=weight_init, l2_lambda=reg_value,
                                 dropout_rate=0)
    elif reg_type == "dropout_rate":
        nn_model = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate,
                                 activation_function, weight_init=weight_init, l2_lambda=0,
                                 dropout_rate=reg_value)
    else:
        nn_model = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate,
                                 activation_function, weight_init=weight_init)

    # Track metrics
    epoch_losses = []
    epoch_auc_scores = []

    for e in range(epochs):
        epoch_loss = 0
        for i in range(len(train_inputs)):
            inputs = train_inputs[i]
            targets = np.zeros(output_nodes) + 0.01
            targets[train_labels[i]] = 0.99
            loss = nn_model.train(inputs, targets)
            epoch_loss += loss
        avg_epoch_loss = epoch_loss / len(train_inputs)
        epoch_losses.append(avg_epoch_loss)

        auc_score = compute_auc(nn_model, test_inputs, test_labels)
        epoch_auc_scores.append(auc_score)

    # Evaluate performance
    scorecard = []
    for i in range(len(test_inputs)):
        correct_label = test_labels[i]
        inputs = test_inputs[i]
        outputs = nn_model.predict(inputs)
        label = np.argmax(outputs)
        scorecard.append(1 if label == correct_label else 0)

    performance = (np.asarray(scorecard).sum() / len(test_inputs)) * 100
    return epoch_losses, epoch_auc_scores, performance


# Test for activation functions with random weight initialization and no regularization
def test_activation_functions():
    results = {}
    for activation_function in activation_functions:
        print(f"\nTesting activation function: {activation_function}")
        epoch_losses, epoch_auc_scores, performance = train_and_evaluate(activation_function, "random")
        results[activation_function] = {
            "losses": epoch_losses,
            "auc_scores": epoch_auc_scores,
            "performance": performance
        }
    return results


# Test for weight initializations with sigmoid activation and no regularization
def test_weight_initializations():
    results = {}
    for weight_init in weight_initializations:
        print(f"\nTesting weight initialization: {weight_init}")
        epoch_losses, epoch_auc_scores, performance = train_and_evaluate("sigmoid", weight_init)
        results[weight_init] = {
            "losses": epoch_losses,
            "auc_scores": epoch_auc_scores,
            "performance": performance
        }
    return results


# Test for regularizations with sigmoid activation and random weight initialization
def test_regularizations():
    results = {}
    for reg_type, reg_values in regularizations.items():
        for reg_value in reg_values:
            print(f"\nTesting regularization {reg_type}={reg_value}")
            epoch_losses, epoch_auc_scores, performance = train_and_evaluate(
                "sigmoid", "random", reg_type=reg_type, reg_value=reg_value)
            results[(reg_type, reg_value)] = {
                "losses": epoch_losses,
                "auc_scores": epoch_auc_scores,
                "performance": performance
            }
    return results


# Running the tests
activation_results = test_activation_functions()
weight_init_results = test_weight_initializations()
regularization_results = test_regularizations()

# Print summary of results for all tests
print("\nSummary of Activation Function Performance:")
for activation_function, result in activation_results.items():
    print(f"{activation_function}: Performance={result['performance']:.2f}%")

print("\nSummary of Weight Initialization Performance:")
for weight_init, result in weight_init_results.items():
    print(f"{weight_init}: Performance={result['performance']:.2f}%")

print("\nSummary of Regularization Performance:")
for (reg_type, reg_value), result in regularization_results.items():
    print(f"{reg_type}={reg_value}: Performance={result['performance']:.2f}%")

# Plotting results for Activation Functions
plt.figure(figsize=(12, 8))
for activation_function, result in activation_results.items():
    plt.plot(range(1, epochs + 1), result["losses"], label=f"{activation_function} Loss")
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Activation Function Test - Training Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
for activation_function, result in activation_results.items():
    plt.plot(range(1, epochs + 1), result["auc_scores"], label=f"{activation_function} AUC")
plt.xlabel('Epoch')
plt.ylabel('AUC Score')
plt.title('Activation Function Test - AUC Scores')
plt.legend()
plt.show()

# Plotting results for Weight Initializations
plt.figure(figsize=(12, 8))
for weight_init, result in weight_init_results.items():
    plt.plot(range(1, epochs + 1), result["losses"], label=f"{weight_init} Loss")
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Weight Initialization Test - Training Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
for weight_init, result in weight_init_results.items():
    plt.plot(range(1, epochs + 1), result["auc_scores"], label=f"{weight_init} AUC")
plt.xlabel('Epoch')
plt.ylabel('AUC Score')
plt.title('Weight Initialization Test - AUC Scores')
plt.legend()
plt.show()

# Plotting results for Regularizations
plt.figure(figsize=(12, 8))
for (reg_type, reg_value), result in regularization_results.items():
    plt.plot(range(1, epochs + 1), result["losses"], label=f"{reg_type}={reg_value} Loss")
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Regularization Test - Training Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
for (reg_type, reg_value), result in regularization_results.items():
    plt.plot(range(1, epochs + 1), result["auc_scores"], label=f"{reg_type}={reg_value} AUC")
plt.xlabel('Epoch')
plt.ylabel('AUC Score')
plt.title('Regularization Test - AUC Scores')
plt.legend()
plt.show()
